# battery_project/trainers/gan_trainer.py
import tensorflow as tf
import logging
import time
import os
from typing import Dict, Optional, Tuple, Iterator

from .base_trainer import BaseTrainer
# Use ModelRegistry to get builder functions or import directly
from ..models.gan import build_generator, build_discriminator
# from ..models import ModelRegistry # Alternative if registered
from ..utils.metrics import R2Score # Add imports for any specific metrics needed

logger = logging.getLogger(__name__)


class GradientHandler:
    """Handles gradient accumulation logic."""
    def __init__(self, accumulation_steps=1):
        self.accumulation_steps = tf.constant(accumulation_steps, dtype=tf.int32)
        self.step = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.accumulated_gradients = []

    def accumulate_gradients(self, train_step_fn, *args):
        """Accumulates gradients for one logical step."""
        # Execute the training step function within a gradient tape
        with tf.GradientTape() as tape:
            loss = train_step_fn(*args) # Call the specific train step (G or D)

        # Get gradients for the trainable variables involved in the step
        # Determine which variables belong to G or D based on the function or context
        # This needs refinement: how does it know which model's vars to watch?
        # Assume the train_step_fn implicitly uses the correct model (G or D)
        # and the tape watches the appropriate variables.
        variables = args[0].trainable_variables if isinstance(args[0], tf.keras.Model) else None # Heuristic: assume first arg is model
        if variables is None:
             # Try to get model from self if passed differently? Very complex.
             # Let's assume the tape correctly captured vars from the called model.
             # This might require the models (G/D) to be attributes of the class calling this.
             logger.error("Cannot determine trainable variables for gradient accumulation.")
             return loss # Return loss but accumulation won't work reliably


        gradients = tape.gradient(loss, variables)

        if not self.accumulated_gradients:
            # First step, initialize accumulated gradients list
            self.accumulated_gradients = [tf.Variable(tf.zeros_like(grad), trainable=False) for grad in gradients]

        # Accumulate gradients
        for i, grad in enumerate(gradients):
            if grad is not None:
                self.accumulated_gradients[i].assign_add(grad / tf.cast(self.accumulation_steps, grad.dtype))

        self.step.assign_add(1)
        return loss # Return the loss for this micro-step

    def apply_gradients(self, optimizer, variables):
        """Applies accumulated gradients if accumulation steps are reached."""
        tf.debugging.assert_equal(self.step % self.accumulation_steps, 0,
                                   message="apply_gradients called at incorrect step")
        optimizer.apply_gradients(zip(self.accumulated_gradients, variables))
        # Reset accumulated gradients and step count for the next logical step
        self.step.assign(0)
        for grad_var in self.accumulated_gradients:
            grad_var.assign(tf.zeros_like(grad_var))
        self.accumulated_gradients = [] # Clear list

    def should_apply(self):
        """Checks if it's time to apply gradients."""
        return (self.step + 1) % self.accumulation_steps == 0


# --- Revisit GradientHandler ---
# The above GradientHandler has issues determining the correct variables.
# A simpler approach is to accumulate within the GANTrainer directly.
# Let's remove GradientHandler and implement accumulation in the trainer loop.
# -------------------------------


class GANTrainer(BaseTrainer):
    """
    Trainer specifically designed for Conditional Generative Adversarial Networks (CGANs),
    particularly Wasserstein GAN with Gradient Penalty (WGAN-GP).
    Handles the alternating training of Generator and Discriminator.
    """
    def __init__(self, config: Dict, model: tf.keras.Model = None, hardware_adapter=None):
        # Note: GANs typically don't use a single 'model', but a G and D pair.
        # We build them internally based on config.
        super().__init__(config, model=None, hardware_adapter=hardware_adapter) # Pass model=None

        self.gan_config = config.get("model", {}).get("gan", {}) # GAN specific model params
        self.trainer_gan_config = config.get("trainer", {}).get("gan", {}) # GAN specific trainer params

        # --- Model Hyperparameters ---
        self.noise_dim = self.gan_config.get("noise_dim", 128)
        self.cond_dim = self.gan_config.get("cond_dim", 1) # Dimension of conditional input
        # Input shapes needed for building models (try to get from config or data later)
        self.input_seq_len = config.get("data.sequence_length", 100)
        self.input_features = self._get_input_features(config) # Helper to determine feature count

        # --- Trainer Hyperparameters ---
        self.generator_lr = self.trainer_gan_config.get("generator_lr", 1e-4)
        self.discriminator_lr = self.trainer_gan_config.get("discriminator_lr", 1e-4)
        self.discriminator_steps = self.trainer_gan_config.get("discriminator_steps", 5) # n_critic
        self.gp_weight = self.trainer_gan_config.get("gp_weight", 10.0)
        self.g_optimizer_name = self.trainer_gan_config.get("generator_optimizer", "adam")
        self.d_optimizer_name = self.trainer_gan_config.get("discriminator_optimizer", "adam")
        # Adam betas for GANs often set differently (e.g., beta_1=0.5)
        self.adam_beta_1 = self.trainer_gan_config.get("adam_beta_1", 0.5)
        self.adam_beta_2 = self.trainer_gan_config.get("adam_beta_2", 0.9) # Usually 0.9 or 0.999

        # --- Condition Source ---
        # Determine how to extract condition data from input batch (features, target)
        self.condition_source = self.trainer_gan_config.get("condition_source", "target") # 'target' or 'feature_index'
        if self.condition_source == 'feature_index':
             # Index of the feature column containing the condition
             self.condition_feature_index = self.trainer_gan_config.get("condition_feature_index", -1)
             logger.info(f"GAN condition source: Feature at index {self.condition_feature_index}")
        else:
             logger.info(f"GAN condition source: Target variable (y)")

        # Optional auxiliary conditional loss weight
        self.condition_loss_weight = tf.constant(
            self.trainer_gan_config.get("condition_loss_weight", 0.0), dtype=tf.float32
        )


        # Build Generator and Discriminator
        self.generator = None
        self.discriminator = None
        self.g_optimizer = None
        self.d_optimizer = None
        self.build_models() # Build G and D here

        # Gradient Accumulation Setup (if needed, simpler implementation)
        self.g_grad_accum_steps = config.get("trainer.gradient_accumulation_steps", 1)
        self.d_grad_accum_steps = self.g_grad_accum_steps # Use same for simplicity or configure separately
        self.g_accumulated_gradients = None
        self.d_accumulated_gradients = None
        self.g_step_in_batch = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.d_step_in_batch = tf.Variable(0, dtype=tf.int32, trainable=False)

        # No primary 'model' attribute from BaseTrainer needed here
        self.model = None # Override model from BaseTrainer

    def _get_input_features(self, config: Dict) -> int:
        """Helper to determine the number of input features based on config."""
        base_features = config.get("data.feature_columns", [])
        num_base = len(base_features)
        num_added = 0
        if config.get("data.extract_health_features", False):
            # Estimate added features based on FormatParser logic (this is still approximate)
            feature_cfg = config.get("data.features", {})
            added_feature_names = []
            if feature_cfg.get("ic.enabled", True): added_feature_names.extend([f'ic_peak{i+1}_{v}' for i in range(2) for v in ['v', 'q']] + ['ic_avg_dqdv'])
            if feature_cfg.get("dtv.enabled", True): added_feature_names.extend([f'dtv_peak{i+1}_{v}' for i in range(2) for v in ['v', 't']] + ['dtv_avg_dtdv'])
            if feature_cfg.get("ctv.enabled", False): added_feature_names.append('ctv_avg_dvdt')
            num_added = len(set(added_feature_names)) # Count unique potential features

        total_features = num_base + num_added
        if total_features == 0:
             raise ValueError("Could not determine number of input features for GAN models.")
        logger.info(f"Determined input features for GAN: {total_features} (Base: {num_base}, Added: {num_added})")
        return total_features

    def build_models(self):
        """Builds the Generator and Discriminator models."""
        logger.info("Building Generator and Discriminator models...")
        g_cfg = self.gan_config.get("generator", {})
        d_cfg = self.gan_config.get("discriminator", {})

        # Use self.strategy scope if in distributed training context (from BaseTrainer)
        with self.strategy.scope():
            self.generator = build_generator(
                noise_dim=self.noise_dim,
                cond_dim=self.cond_dim,
                output_seq_len=self.input_seq_len,
                output_features=self.input_features,
                cfg=g_cfg
            )
            self.discriminator = build_discriminator(
                input_seq_len=self.input_seq_len,
                input_features=self.input_features,
                cond_dim=self.cond_dim,
                cfg=d_cfg
            )
            logger.info("Generator and Discriminator built.")
            # Build optimizers here as well
            self.g_optimizer = self.get_optimizer(
                self.g_optimizer_name, self.generator_lr, beta_1=self.adam_beta_1, beta_2=self.adam_beta_2
            )
            self.d_optimizer = self.get_optimizer(
                self.d_optimizer_name, self.discriminator_lr, beta_1=self.adam_beta_1, beta_2=self.adam_beta_2
            )
            # Apply mixed precision policy if enabled
            if self.use_mixed_precision:
                 self.g_optimizer = tf.keras.mixed_precision.LossScaleOptimizer(self.g_optimizer)
                 self.d_optimizer = tf.keras.mixed_precision.LossScaleOptimizer(self.d_optimizer)
            logger.info("GAN Optimizers created.")


    def compile_model(self):
        """GANs compile internally; this method might not be needed or used differently."""
        # GANs don't typically use model.compile() in the same way.
        # Loss is calculated manually, metrics updated manually.
        logger.warning("GANTrainer does not use standard model.compile(). Optimizers and models built in __init__.")
        pass # Override compile_model from BaseTrainer


    def get_metrics(self) -> Dict[str, tf.keras.metrics.Metric]:
        """Returns metrics for GAN training (e.g., G/D loss)."""
        # Define metrics relevant to GAN training
        return {
            'g_loss': tf.keras.metrics.Mean(name='g_loss'),
            'd_loss': tf.keras.metrics.Mean(name='d_loss'),
            # Add other metrics like Gradient Penalty magnitude, FID score (needs external calculation) etc.
            'gp_metric': tf.keras.metrics.Mean(name='gp_metric') # Example for GP
        }

    @tf.function
    def _gradient_penalty(self, real_sequences, fake_sequences, condition_inputs):
         """Calculates the gradient penalty loss for WGAN-GP."""
         batch_size = tf.shape(real_sequences)[0]
         seq_len = tf.shape(real_sequences)[1]
         num_features = tf.shape(real_sequences)[2]

         # Generate random interpolation factor
         alpha = tf.random.uniform([batch_size, 1, 1], 0., 1.) # Shape (batch, 1, 1) for broadcasting
         # Interpolate between real and fake sequences
         interpolated_sequences = real_sequences + alpha * (fake_sequences - real_sequences)

         with tf.GradientTape() as gp_tape:
             gp_tape.watch(interpolated_sequences)
             # Get Discriminator output for interpolated sequences AND conditions
             pred = self.discriminator([interpolated_sequences, condition_inputs], training=True)

         # Calculate gradients of D output with respect to interpolated inputs
         grads = gp_tape.gradient(pred, [interpolated_sequences])[0]
         if grads is None:
              tf.print("Warning: Gradients for Gradient Penalty are None!", output_stream=sys.stderr)
              return tf.constant(0.0, dtype=tf.float32)

         # Calculate the norm of the gradients
         norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2])) # Sum over seq_len and features

         # Calculate gradient penalty (L2 norm penalty)
         gp = tf.reduce_mean((norm - 1.0) ** 2)
         return gp

    # --- Train Step Functions (internal logic for G and D) ---
    # These functions will be called by the main train loop and potentially GradientHandler

    def _generator_train_step_logic(self, noise, condition_inputs):
        """Internal logic for one generator step, returns loss for accumulation."""
        with tf.GradientTape() as tape:
            fake_sequences = self.generator([noise, condition_inputs], training=True)
            # Get Discriminator's score for fake sequences (no D training here)
            fake_output = self.discriminator([fake_sequences, condition_inputs], training=False)
            # Calculate generator loss (maximizes D score for fakes, hence negative)
            g_loss = -tf.reduce_mean(fake_output)

            # --- Optional: Conditional Loss Placeholder ---
            if self.condition_loss_weight > 0.0:
                 # Add auxiliary head to generator if not already present
                 # Need to modify build_generator or add head here temporarily (less ideal)
                 # predicted_condition = self.generator.auxiliary_head(fake_sequences) # Assuming head exists
                 # condition_loss = tf.reduce_mean(tf.square(predicted_condition - condition_inputs))
                 # g_loss += self.condition_loss_weight * condition_loss
                 tf.print("Warning: Conditional loss not implemented yet.", output_stream=sys.stderr)
            # --------------------------------------------

            # Apply mixed precision scaling to the loss if enabled
            scaled_g_loss = self.g_optimizer.get_scaled_loss(g_loss) if self.use_mixed_precision else g_loss

        # Get gradients of scaled loss w.r.t. generator variables
        scaled_gradients = tape.gradient(scaled_g_loss, self.generator.trainable_variables)
        return scaled_g_loss, scaled_gradients


    def _discriminator_train_step_logic(self, real_sequences, noise, condition_inputs):
        """Internal logic for one discriminator step, returns loss and GP for accumulation."""
        with tf.GradientTape() as tape:
            # Generate fake sequences (no G training here)
            fake_sequences = self.generator([noise, condition_inputs], training=False)

            # Get Discriminator scores for real and fake sequences
            real_output = self.discriminator([real_sequences, condition_inputs], training=True)
            fake_output = self.discriminator([fake_sequences, condition_inputs], training=True)

            # Calculate WGAN discriminator cost (D(real) - D(fake)) - we want to maximize this
            d_cost = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)

            # Calculate gradient penalty
            gp = self._gradient_penalty(real_sequences, fake_sequences, condition_inputs)

            # Total discriminator loss
            d_loss = d_cost + gp * self.gp_weight

            # Apply mixed precision scaling if enabled
            scaled_d_loss = self.d_optimizer.get_scaled_loss(d_loss) if self.use_mixed_precision else d_loss

        # Get gradients of scaled loss w.r.t. discriminator variables
        scaled_gradients = tape.gradient(scaled_d_loss, self.discriminator.trainable_variables)
        return scaled_d_loss, scaled_gradients, gp # Return GP for logging


    # --- Main Training Loop ---
    def train(self, train_data: tf.data.Dataset, val_data: Optional[tf.data.Dataset] = None, epochs: Optional[int] = None, train_steps: Optional[int] = None, val_steps: Optional[int] = None, callbacks: Optional[List[tf.keras.callbacks.Callback]] = None):
        """
        Trains the GAN model (Generator and Discriminator).

        Args:
            train_data: Training dataset iterator.
            val_data: Validation dataset iterator (optional, used by callbacks).
            epochs: Number of epochs to train (overrides config if provided).
            train_steps: Number of steps per training epoch (optional).
            val_steps: Number of steps per validation epoch (optional).
            callbacks: List of Keras callbacks.
        """
        target_epochs = epochs if epochs is not None else self.epochs
        callbacks = callbacks or self.get_default_callbacks() # Get default callbacks if none provided
        history = tf.keras.callbacks.History() # Use Keras History object
        callback_list = tf.keras.callbacks.CallbackList(
            callbacks + [history], # Add history callback
            add_history=False, # Already added
            model=self, # Pass trainer as model for callbacks
            epochs=target_epochs,
            verbose=1
        )

        # Calculate steps if not provided (requires dataset cardinality)
        if train_steps is None: train_steps = self._estimate_steps(train_data, 'training')
        if val_steps is None: val_steps = self._estimate_steps(val_data, 'validation')
        if train_steps is None:
             logger.error("Cannot determine training steps per epoch. Aborting.")
             return None # Or raise error

        logger.info(f"Starting GAN training for {target_epochs} epochs...")
        logger.info(f"Generator LR: {self.generator_lr}, Discriminator LR: {self.discriminator_lr}")
        logger.info(f"Discriminator steps per Generator step: {self.discriminator_steps}")
        logger.info(f"Gradient Penalty weight: {self.gp_weight}")
        logger.info(f"Gradient Accumulation Steps (G/D): {self.g_grad_accum_steps}")

        # Get metrics managed by BaseTrainer._configure_metrics
        metrics = self._configure_metrics()

        initial_epoch = self._get_initial_epoch(callbacks) # Handle resuming from checkpoint

        callback_list.on_train_begin()

        for epoch in range(initial_epoch, target_epochs):
            epoch_logs = {}
            callback_list.on_epoch_begin(epoch, logs=epoch_logs)

            # Reset metrics at the start of each epoch
            for metric in metrics.values():
                metric.reset_states()
            self.g_step_in_batch.assign(0) # Reset grad accum step counters
            self.d_step_in_batch.assign(0)
            self.g_accumulated_gradients = None # Reset accumulators
            self.d_accumulated_gradients = None


            # Training loop for one epoch
            for step, batch_data in self.track_progress(train_data.take(train_steps), total=train_steps, description=f"Epoch {epoch+1}/{target_epochs}"):
                callback_list.on_train_batch_begin(step, logs={})

                # --- Extract Real Sequences and Conditions ---
                # Assuming batch_data is (features, target) from data_provider
                # Adjust based on actual return format of parser.parse_tfrecord_fn
                real_sequences, y_batch = batch_data

                if self.condition_source == 'target':
                     condition_inputs = y_batch
                elif self.condition_source == 'feature_index':
                     # Assumes condition is static within the sequence
                     condition_inputs = real_sequences[:, 0, self.condition_feature_index : self.condition_feature_index + self.cond_dim]
                     # Remove condition feature from real_sequences if needed? Depends on model design.
                     # real_sequences = tf.concat([real_sequences[:,:,:self.condition_feature_index],
                     #                           real_sequences[:,:,self.condition_feature_index+self.cond_dim:]], axis=-1)
                else:
                     raise ValueError(f"Invalid condition_source: {self.condition_source}")

                # Ensure condition_inputs has the correct shape [batch_size, cond_dim]
                # Remove sequence dimension if present and source is target/static feature
                if len(condition_inputs.shape) > 2:
                     condition_inputs = tf.squeeze(condition_inputs, axis=1)
                # Ensure feature dimension exists if it was scalar
                if len(condition_inputs.shape) == 1:
                    condition_inputs = tf.expand_dims(condition_inputs, axis=-1)
                # Final check
                tf.debugging.assert_shapes([
                    (condition_inputs, ('B', self.cond_dim)),
                ], message="Condition input shape mismatch.")
                # -----------------------------------------------

                batch_size = tf.shape(real_sequences)[0]
                noise = tf.random.normal([batch_size, self.noise_dim])

                # === Train Discriminator (n_critic steps) ===
                total_d_loss_micro = tf.constant(0.0, dtype=tf.float32)
                total_gp_micro = tf.constant(0.0, dtype=tf.float32)
                for _ in range(self.discriminator_steps):
                     # --- Accumulate Discriminator Gradients ---
                     self.d_step_in_batch.assign_add(1)
                     scaled_d_loss, scaled_d_gradients, gp = self._discriminator_train_step_logic(
                         real_sequences, noise, condition_inputs
                     )
                     total_d_loss_micro += scaled_d_loss # Accumulate scaled loss for logging
                     total_gp_micro += gp

                     if self.d_accumulated_gradients is None:
                          self.d_accumulated_gradients = [tf.Variable(tf.zeros_like(g), trainable=False) for g in scaled_d_gradients if g is not None]
                     for i, grad in enumerate(scaled_d_gradients):
                          if grad is not None:
                               self.d_accumulated_gradients[i].assign_add(grad / tf.cast(self.d_grad_accum_steps * self.discriminator_steps, grad.dtype))

                     # Apply gradients every d_grad_accum_steps * discriminator_steps micro-steps
                     if self.d_step_in_batch % (self.d_grad_accum_steps * self.discriminator_steps) == 0:
                          # Unscale gradients if using mixed precision
                          unscaled_d_gradients = self.d_optimizer.get_unscaled_gradients(self.d_accumulated_gradients) if self.use_mixed_precision else self.d_accumulated_gradients
                          # Clip gradients (optional)
                          if self.config.get("trainer.clip_grad_norm"):
                               unscaled_d_gradients, _ = tf.clip_by_global_norm(unscaled_d_gradients, self.config["trainer.clip_grad_norm"])
                          # Apply gradients
                          self.d_optimizer.apply_gradients(zip(unscaled_d_gradients, self.discriminator.trainable_variables))
                          # Reset accumulators
                          self.d_accumulated_gradients = None
                          self.d_step_in_batch.assign(0) # Reset counter for D phase

                # Average loss over micro-steps for logging
                avg_d_loss = total_d_loss_micro / tf.cast(self.discriminator_steps, tf.float32)
                avg_gp = total_gp_micro / tf.cast(self.discriminator_steps, tf.float32)

                # === Train Generator ===
                # --- Accumulate Generator Gradients ---
                self.g_step_in_batch.assign_add(1)
                scaled_g_loss, scaled_g_gradients = self._generator_train_step_logic(
                    noise, condition_inputs
                )

                if self.g_accumulated_gradients is None:
                     self.g_accumulated_gradients = [tf.Variable(tf.zeros_like(g), trainable=False) for g in scaled_g_gradients if g is not None]
                for i, grad in enumerate(scaled_g_gradients):
                     if grad is not None:
                          self.g_accumulated_gradients[i].assign_add(grad / tf.cast(self.g_grad_accum_steps, grad.dtype))

                # Apply gradients every g_grad_accum_steps micro-steps
                if self.g_step_in_batch % self.g_grad_accum_steps == 0:
                     unscaled_g_gradients = self.g_optimizer.get_unscaled_gradients(self.g_accumulated_gradients) if self.use_mixed_precision else self.g_accumulated_gradients
                     if self.config.get("trainer.clip_grad_norm"):
                          unscaled_g_gradients, _ = tf.clip_by_global_norm(unscaled_g_gradients, self.config["trainer.clip_grad_norm"])
                     self.g_optimizer.apply_gradients(zip(unscaled_g_gradients, self.generator.trainable_variables))
                     self.g_accumulated_gradients = None
                     self.g_step_in_batch.assign(0) # Reset counter for G phase


                # Update metrics (use unscaled losses)
                # Need to unscale loss manually if optimizers don't provide it directly
                # Placeholder: Log scaled loss for now
                metrics['d_loss'].update_state(avg_d_loss)
                metrics['g_loss'].update_state(scaled_g_loss)
                metrics['gp_metric'].update_state(avg_gp)

                batch_logs = {m.name: m.result() for m in metrics.values()}
                callback_list.on_train_batch_end(step, logs=batch_logs)


            # End of epoch
            epoch_logs.update({m.name: m.result() for m in metrics.values()})

            # --- Validation Phase (Optional for GANs) ---
            # GANs often don't have a standard validation loop like supervised models.
            # Validation might involve generating samples and calculating metrics like FID or IS.
            # For simplicity, we can skip standard validation loss/metrics calculation here.
            # Callbacks might still perform validation actions (e.g., generating samples).
            if val_data and val_steps:
                 logger.info("Running validation phase actions (e.g., callbacks)...")
                 # Callbacks handle validation logic if needed (e.g., SampleGenerator callback)
                 callback_list.on_test_begin(logs=epoch_logs)
                 # Minimal validation: run one batch to allow callbacks to potentially use val data
                 val_batch_example = next(iter(val_data.take(1)))
                 callback_list.on_test_batch_begin(0, logs={})
                 # Add dummy val metrics if callbacks expect them
                 epoch_logs['val_loss'] = 0.0
                 epoch_logs['val_g_loss'] = 0.0
                 epoch_logs['val_d_loss'] = 0.0
                 callback_list.on_test_batch_end(0, logs=epoch_logs)
                 callback_list.on_test_end(logs=epoch_logs)
            # -----------------------------------------

            callback_list.on_epoch_end(epoch, logs=epoch_logs)

            # Check for early stopping (if callback included)
            if hasattr(self, 'stop_training') and self.stop_training:
                 logger.info(f"Early stopping triggered at epoch {epoch+1}.")
                 break

        callback_list.on_train_end(logs=epoch_logs)
        logger.info("GAN training finished.")
        return history # Return Keras history object

    def predict(self, input_data: Union[tf.data.Dataset, Tuple[np.ndarray, np.ndarray]], steps: Optional[int] = None) -> np.ndarray:
        """Generates sequences using the generator model."""
        logger.info("Generating samples using the generator...")
        all_generated_sequences = []

        # Input data should provide noise and conditions
        if isinstance(input_data, tf.data.Dataset):
             iterator = iter(input_data)
             if steps is None:
                  steps = self._estimate_steps(input_data, "prediction")
             if steps is None:
                  logger.warning("Cannot determine steps for prediction dataset. Predicting on first batch only.")
                  steps = 1
             for _ in tqdm(range(steps), desc="Generating"):
                  try:
                      noise, condition = next(iterator) # Assuming dataset yields (noise, condition)
                      generated = self.generator([noise, condition], training=False)
                      all_generated_sequences.append(generated.numpy())
                  except StopIteration:
                      break
        elif isinstance(input_data, tuple) and len(input_data) == 2:
             noise, condition = input_data
             # Add batch dimension if single sample
             if len(noise.shape) == 1: noise = tf.expand_dims(noise, axis=0)
             if len(condition.shape) == 1: condition = tf.expand_dims(condition, axis=0)
             generated = self.generator([noise, condition], training=False)
             all_generated_sequences.append(generated.numpy())
        else:
             logger.error("Invalid input data format for prediction. Provide Dataset or (noise, condition) tuple.")
             return np.array([])

        if not all_generated_sequences:
             return np.array([])

        return np.concatenate(all_generated_sequences, axis=0)

    # Override other methods from BaseTrainer if needed (e.g., save/load for G and D separately)
    # def save_model(...)
    # def load_model(...)