# Tasks: Human Teaching Mode

## 1. Core Implementation

- [x] 1.1 Create `examples/franka/keyboard_utils.py` with `cbreak_terminal()` and `check_key_pressed()`
- [x] 1.2 Add teaching config fields to `RealEnvConfig` in `real_env.py`
- [x] 1.3 Update `RealEnvConfig.from_yaml()` to load teaching config from YAML
- [x] 1.4 Add `teaching` section to `real_env_config.yaml`
- [x] 1.5 Implement `FrankaRealEnv.enable_teaching_mode()` method
- [x] 1.6 Add `FrankaRealEnv.is_teaching_mode` property
- [x] 1.7 Add `_teaching_mode = False` initialization in `FrankaRealEnv.__init__()`
- [x] 1.8 Reset `_teaching_mode` in `FrankaRealEnv.reset()`

## 2. Environment Integration

- [x] 2.1 Add `_teaching_mode_active` and `_keyboard_enabled` to `FrankaEnvironment.__init__()`
- [x] 2.2 Implement `FrankaEnvironment.is_teaching_mode` property
- [x] 2.3 Implement `FrankaEnvironment.enable_teaching_mode()` method
- [x] 2.4 Implement `FrankaEnvironment._check_teaching_trigger()` method
- [x] 2.5 Add `_check_teaching_trigger()` call at start of `apply_action()`
- [x] 2.6 Modify `is_episode_complete()` to skip timeout when `_teaching_mode_active`
- [x] 2.7 Reset `_teaching_mode_active` in `FrankaEnvironment.reset()`

## 3. Recording Integration

- [x] 3.1 Add `is_human_teaching` field to `_build_record()` in `pkl_recorder.py`

## 4. Main Entry Point

- [x] 4.1 Import `cbreak_terminal` in `main.py`
- [x] 4.2 Wrap `runtime.run()` with `cbreak_terminal()` context manager
- [x] 4.3 Fix: Set `max_episode_steps=0` to disable Runtime step limit (timeout handled by FrankaEnvironment)

## 5. Validation

- [x] 5.1 Manual test: press spacebar during impedance evaluation, verify zero resistance
- [x] 5.2 Manual test: verify `step=N` continues incrementing after teaching mode
- [x] 5.3 Manual test: verify PKL contains `is_human_teaching` field
- [x] 5.4 Manual test: verify Ctrl+C saves episode correctly
- [x] 5.5 Manual test: verify timeout is disabled after teaching mode
- [x] 5.6 Manual test: verify idempotency (multiple spacebar presses)
