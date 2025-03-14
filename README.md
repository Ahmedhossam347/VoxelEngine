# Retro Voxel Engine

## Overview
This voxel engine is inspired by the retro voxel rendering techniques used in 90s games like *Comanche*.
![Voxel Engine Preview](images/voxel_preview.png)
## Features
- **Optimized for Performance:**
  - Easier to run on older hardware.
  - Lower RAM usage compared to modern voxel engines.
  - Uses pre-rendered shadows and maps for efficient rendering.

- **Dynamic World Rendering:**
  - Infinite or limited world size.
  - Chunk-based loading for seamless exploration.
  - Ray-casting for realistic terrain visualization.

- **Player Mechanics:**
  - Smooth movement with inertia and damping.
  - Adjustable field of view and draw distance.
  - Flying mode and gravity simulation.

- **Graphical Enhancements:**
  - Sky gradient with dynamic sun rendering.
  - Terrain height-based shading.
  - Realistic pitch and yaw adjustments.

## Configuration
- **World Settings:**
  - `CHUNK_SIZE`: Defines the size of a chunk.
  - `LOAD_DISTANCE`: Controls the number of chunks loaded around the player.
  - `DRAW_DISTANCE`: Determines how far the player can see.
  - `INFINITE_MAP`: Enables or disables infinite terrain.

- **Player Physics:**
  - `ACCELERATION`: Adjusts movement speed.
  - `DAMPING`: Controls movement inertia.
  - `PITCH_ACCEL` and `PITCH_DAMPING`: Adjust pitch sensitivity.
  - `GRAVITY`: Defines the player's gravity.
  - `JUMP_FORCE`: Determines the jump strength.

- **Rendering Options:**
  - `WORLD_SCALE`: Adjusts the size of the player relative to the world.
  - `BREATH_AMPLITUDE` and `BREATH_FREQUENCY`: Simulates breathing motion.
  - `PITCH_SKY_FACTOR`: Defines sky movement sensitivity.

## Installation & Usage
1. Install dependencies:
   ```bash
   pip install pygame numba numpy
   ```
2. Run the engine:
   ```bash
   python main.py
   ```
3. Controls:
   - `WASD` to move.
   - `Mouse` to look around.
   - `SPACE` to jump/fly.
   - `SHIFT` to descend while flying.
   - `ESC` to exit.

## Performance Notes
- Uses **Numba** for JIT-optimized computations.
- Multi-threaded chunk management for better loading efficiency.
- Adaptive LOD based on draw distance to optimize rendering.

## Future Enhancements
- **Improved AI pathfinding for NPCs.**
- **Procedural terrain generation.**
- **Day/night cycle and weather effects.**

