# main

Entry point of the application. Run via:

```bash
python -m src.main <command> [args]
```

## Commands

| Command | Arguments | Description |
|---|---|---|
| `map-elites` | — | Run map-elites to generate a diverse archive of CPG gaits |
| `train_network` | — | Train the neural network controller via PPO |
| `train_network_pretrain` | `<path/to/archive>` | Pretrain via Behavioral Cloning from a map-elites archive, then fine-tune with PPO |
| `simulate` | `<path/to/controller>` | Run an interactive keyboard-controlled simulation |
| `simulate_controller` | `<path/to/controller>` | Run an interactive joystick-controlled simulation |
| `render` | `<path/to/controller>` | Render a video of the trained controller |
