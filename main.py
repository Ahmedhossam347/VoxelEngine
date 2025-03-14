import pygame as pg
from numba import njit
import numpy as np
import math
from threading import Thread
import queue

# Configuration constants
CHUNK_SIZE = 10
LOAD_DISTANCE = 1
DRAW_DISTANCE = 5000
INFINITE_MAP = False
# Sky breathing effect configuration
SKY_AMPLITUDE = 10      # Maximum vertical shift (in pixels) for the sky
SKY_FREQUENCY = 0.0005  # Oscillation frequency for the sky

# Player breathing effect
BREATH_AMPLITUDE = 2    # Maximum vertical "breath" offset
BREATH_FREQUENCY = 0.002  # Frequency of the breathing effect

# Movement inertia configuration
ACCELERATION = 2 # How fast velocity builds up
DAMPING = 0.8      # Velocity decay factor each frame

# Pitch (look up/down) control configuration
PITCH_ACCEL = 0.5   # How fast pitch velocity builds
PITCH_DAMPING = 0.9 # Damping for pitch velocity
PITCH_MIN, PITCH_MAX = -1000, 1000  # Limits for pitch (in degrees)


# Sky offset sensitivity to pitch (pixels per degree)
PITCH_SKY_FACTOR = 0.1

# World scale configuration
WORLD_SCALE = 1 # Default scale: 1.0 = human size, <1 = giant world, >1 = tiny world
WORLD_SCALE_MIN = 1  # Minimum scale (very large world)
WORLD_SCALE_MAX = 5 # Maximum scale (very small world)
WORLD_SCALE_STEP = 0.1 # How much to change scale per key press
# Load base height map and color map
base_height_map = pg.surfarray.array3d(pg.image.load('D1.png'))
base_color_map = pg.surfarray.array3d(pg.image.load('C1W.png'))

# ------------------------------------------------------------
# Chunk and Chunk Manager Functions
# ------------------------------------------------------------
def create_chunk(x, y):
    return {'x': x, 'y': y, 'heightmap': None, 'colormap': None}

def chunk_loader(cm):
    while True:
        chunk_coord = cm['load_queue'].get()
        if chunk_coord not in cm['loaded_chunks']:
            cm['loaded_chunks'][chunk_coord] = create_chunk(*chunk_coord)
        cm['load_queue'].task_done()

def init_chunk_manager():
    cm = {}
    cm['loaded_chunks'] = {}
    cm['active_area'] = (0, 0)
    cm['load_queue'] = queue.Queue()
    worker = Thread(target=chunk_loader, args=(cm,))
    worker.daemon = True
    worker.start()
    cm['worker'] = worker
    return cm

def get_chunk(world_x, world_y):
    chunk_x = world_x // CHUNK_SIZE
    chunk_y = world_y // CHUNK_SIZE
    return (chunk_x, chunk_y)

def load_chunks_around(player_pos, cm):
    current_chunk = get_chunk(player_pos[0], player_pos[1])
    chunks_to_load = set()
    for dx in range(-LOAD_DISTANCE, LOAD_DISTANCE + 1):
        for dy in range(-LOAD_DISTANCE, LOAD_DISTANCE + 1):
            chunk_coord = (current_chunk[0] + dx, current_chunk[1] + dy)
            chunks_to_load.add(chunk_coord)
    for chunk in list(cm['loaded_chunks'].keys()):
        if chunk not in chunks_to_load:
            del cm['loaded_chunks'][chunk]
    for chunk_coord in chunks_to_load:
        if chunk_coord not in cm['loaded_chunks']:
            cm['load_queue'].put(chunk_coord)

# ------------------------------------------------------------
# Numba-accelerated Functions
# ------------------------------------------------------------
@njit(cache=True, fastmath=True)
def optimized_collision_check(pos, height_map):
    if INFINITE_MAP:
        x = int(pos[0]) % height_map.shape[0]
        y = int(pos[1]) % height_map.shape[1]
    else:
        x = int(pos[0])
        y = int(pos[1])
        if x < 0:
            x = 0
        elif x >= height_map.shape[0]:
            x = height_map.shape[0] - 1
        if y < 0:
            y = 0
        elif y >= height_map.shape[1]:
            y = height_map.shape[1] - 1
    return height_map[x, y][0]

@njit(fastmath=True)
def ray_casting(screen_array, player_pos, player_angle, player_height, player_pitch,
                screen_width, screen_height, delta_angle, ray_distance, h_fov, scale_height, world_scale):
    y_buffer = np.full(screen_width, screen_height)
    map_width = base_height_map.shape[0]
    map_height = base_height_map.shape[1]
    ray_angle = player_angle - h_fov
    for num_ray in range(screen_width):
        first_contact = False
        sin_a = math.sin(ray_angle)
        cos_a = math.cos(ray_angle)
        for depth in range(1, ray_distance):
            if INFINITE_MAP:
                x = int(player_pos[0] + depth * cos_a) % map_width
                y = int(player_pos[1] + depth * sin_a) % map_height
            else:
                x = int(player_pos[0] + depth * cos_a)
                y = int(player_pos[1] + depth * sin_a)
                if x < 0 or x >= map_width or y < 0 or y >= map_height:
                    break
            depth_corr = depth * math.cos(player_angle - ray_angle)
            # Scale terrain height and player height by world_scale
            scaled_height = base_height_map[x, y][0] * world_scale
            height_on_screen = int((player_height - scaled_height) /
                                 depth_corr * scale_height + player_pitch)
            if not first_contact:
                y_buffer[num_ray] = min(height_on_screen, screen_height)
                first_contact = True
            if height_on_screen < 0:
                height_on_screen = 0
            if height_on_screen < y_buffer[num_ray]:
                for screen_y in range(height_on_screen, y_buffer[num_ray]):
                    screen_array[num_ray, screen_y] = base_color_map[x, y]
                y_buffer[num_ray] = height_on_screen
        ray_angle += delta_angle
    return screen_array

def create_player():
    initial_pos = np.array([base_height_map.shape[0] // 2, base_height_map.shape[1] // 2], dtype=float)
    terrain_height = optimized_collision_check(initial_pos, base_height_map)
    return {
        'pos': initial_pos,      # Player position (X, Y)
        'height': terrain_height + 40,  # Start above terrain
        'vel': np.array([0.0, 0.0]),  # Horizontal velocity (X, Y)
        'v_vel': 0.0,  
        'pitch': 0.0,  # Camera pitch
        'pitch_vel': 0.0,  # Pitch velocity (for smooth camera movement)
        'flying': False,  # Is player in flying mode?
        'last_space_press': 0,  # Timing for double-tap space to toggle flying
        'space_pressed_prev': False,  # Track previous space press
        'angle': math.pi / 4,  # Yaw (horizontal rotation)
        'angle_vel': 0.006,  # Mouse sensitivity for looking around
        'is_on_ground': True,  # Track if player is touching ground
        'velocity_y': 0,  # Gravity & jumping physics
        'gravity': 0.9,  # Gravity force
        'jump_force': 12,  # Jump power
    }

def update_player(player, mouse_rel, draw_distance, cm, dt, world_scale):  # Added world_scale
    keys = pg.key.get_pressed()
    mouse_buttons = pg.mouse.get_pressed()
    current_time = pg.time.get_ticks()

    # --- Mouse Look ---
    player['angle'] += mouse_rel[0] * player['angle_vel'] * dt * 60
    player['pitch'] -= mouse_rel[1] * 0.7 * dt * 60
    player['pitch'] = max(min(player['pitch'], PITCH_MAX), PITCH_MIN)

    # --- Horizontal Movement ---
    input_vec = np.array([0.0, 0.0])
    if keys[pg.K_w] or mouse_buttons[0]:
        input_vec[1] += 1
    if keys[pg.K_s]:
        input_vec[1] -= 1
    if keys[pg.K_a]:
        input_vec[0] -= 1
    if keys[pg.K_d]:
        input_vec[0] += 1
    if np.linalg.norm(input_vec) != 0:
        input_vec = input_vec / np.linalg.norm(input_vec)
    
    # Adjust movement speed based on world_scale (inverse scaling)
    scale_factor = 1.0 / world_scale
    player['vel'] += input_vec * ACCELERATION * dt * 60 * scale_factor
    player['vel'] *= math.pow(DAMPING, dt * 60)
    move_dir = np.array([
        math.cos(player['angle']) * player['vel'][1] - math.sin(player['angle']) * player['vel'][0],
        math.sin(player['angle']) * player['vel'][1] + math.cos(player['angle']) * player['vel'][0]
    ])
    player['pos'] += move_dir * dt * 60 * scale_factor

    # --- Clamp Position ---
    if not INFINITE_MAP:
        map_width = base_height_map.shape[0]
        map_height = base_height_map.shape[1]
        if player['pos'][0] < 0:
            player['pos'][0] = 0
            player['vel'][0] = max(0, player['vel'][0])
        elif player['pos'][0] >= map_width:
            player['pos'][0] = map_width - 1
            player['vel'][0] = min(0, player['vel'][0])
        if player['pos'][1] < 0:
            player['pos'][1] = 0
            player['vel'][1] = max(0, player['vel'][1])
        elif player['pos'][1] >= map_height:
            player['pos'][1] = map_height - 1
            player['vel'][1] = min(0, player['vel'][1])

    # --- Vertical Movement ---
    vertical_input = 0
    if keys[pg.K_SPACE]:
        vertical_input += 1
    if keys[pg.K_LSHIFT] or keys[pg.K_RSHIFT]:
        vertical_input -= 1
    player['v_vel'] += vertical_input * ACCELERATION * dt * 60 * scale_factor
    player['v_vel'] *= math.pow(DAMPING, dt * 60)
    player['height'] += player['v_vel'] * dt * 60 * scale_factor

    # --- Terrain Collision ---
    terrain_height = optimized_collision_check(player['pos'], base_height_map) * world_scale
    if player['height'] < terrain_height + 40 * world_scale:  # Scale player height offset
        player['height'] = terrain_height + 40 * world_scale
        player['v_vel'] = 0

    # --- Pitch ---
    pitch_input = 0
    if keys[pg.K_UP]:
        pitch_input -= 1
    if keys[pg.K_DOWN]:
        pitch_input += 1
    player['pitch_vel'] += pitch_input * PITCH_ACCEL * dt * 60
    player['pitch_vel'] *= math.pow(PITCH_DAMPING, dt * 60)
    player['pitch'] += player['pitch_vel'] * dt * 60
    player['pitch'] = max(min(player['pitch'], PITCH_MAX), PITCH_MIN)

    # --- Flying Toggle ---
    if not player['flying']:
        if keys[pg.K_SPACE] and not player['space_pressed_prev']:
            if current_time - player['last_space_press'] < 300:
                player['flying'] = True
                player['velocity_y'] = 0
            else:
                if player['is_on_ground']:
                    player['velocity_y'] = player['jump_force'] * scale_factor
                    player['is_on_ground'] = False
            player['last_space_press'] = current_time
        player['space_pressed_prev'] = keys[pg.K_SPACE]
    else:
        if keys[pg.K_SPACE] and not player['space_pressed_prev']:
            if current_time - player['last_space_press'] < 300:
                player['flying'] = False
                player['velocity_y'] = 0
            player['last_space_press'] = current_time
        player['space_pressed_prev'] = keys[pg.K_SPACE]

    # --- Gravity ---
    if not player['flying']:
        player['velocity_y'] -= player['gravity'] * dt * 60
        player['height'] += player['velocity_y'] * dt * 60 * scale_factor
        terrain_height = optimized_collision_check(player['pos'], base_height_map) * world_scale
        if player['height'] < terrain_height + 40 * world_scale:
            player['height'] = terrain_height + 40 * world_scale
            player['velocity_y'] = 0
            player['is_on_ground'] = True
        else:
            player['is_on_ground'] = False

    # --- Load Chunks ---
    load_chunks_around(player['pos'], cm)

def create_sky_gradient(height):
    gradient = np.zeros((height, 3), dtype=np.uint8)
    for y in range(height):
        gradient[y] = np.array([135, 206, 235]) * (1 - y/height) + np.array([0, 0, 125]) * (y/height)
    return gradient
def create_voxel_render(width, height):
    vr = {}
    vr['fov'] = math.pi / 6
    vr['h_fov'] = vr['fov'] / 2
    vr['num_rays'] = width
    vr['delta_angle'] = vr['fov'] / width
    vr['ray_distance'] = 1000
    vr['scale_height'] = 920

    # Increase sky dimensions to avoid duplicate suns
    sky_width = int(width * 1.5)
    sky_height = int(height * 2)

    # 1) Create the sky gradient surface
    sky_surface = pg.Surface((sky_width, sky_height), pg.SRCALPHA)
    for y in range(sky_height):
        factor = 1 - (y / sky_height)
        sky_color = (
            int(135 * factor),
            int(206 * factor),
            int(235 * factor)
        )
        pg.draw.line(sky_surface, sky_color, (0, y), (sky_width, y))

    # 2) Create a separate sun surface with alpha
    sun_glow_radius = 80
    sun_radius = 50
    sun_size = sun_glow_radius * 2
    sun_surface = pg.Surface((sun_size, sun_size), pg.SRCALPHA)
    sun_surface.fill((0, 0, 0, 0))  # fully transparent background

    # 3) Radial gradient: center = bright yellow, outer = alpha=0
    center = (sun_glow_radius, sun_glow_radius)
    for y in range(sun_size):
        for x in range(sun_size):
            dist = math.hypot(x - center[0], y - center[1])
            if dist <= sun_radius:
                # fully opaque sun core
                sun_surface.set_at((x, y), (255, 255, 0, 255))
            elif dist <= sun_glow_radius:
                # glow zone
                glow_factor = 1 - (dist - sun_radius) / (sun_glow_radius - sun_radius)
                # alpha goes from 255 near sun_radius to 0 at sun_glow_radius
                alpha = int(255 * glow_factor)
                # color is always yellow, just less alpha
                sun_surface.set_at((x, y), (255, 255, 0, alpha))
            else:
                # outside glow
                sun_surface.set_at((x, y), (0, 0, 0, 0))

    # 4) Blit sun onto the sky with normal alpha blend (no special_flags)
    sun_x = sky_width // 2
    sun_y = sky_height // 6
    sky_surface.blit(sun_surface, (sun_x - sun_glow_radius, sun_y - sun_glow_radius))

    # Store pre-rendered sky
    vr['sky_image'] = sky_surface
    vr['screen_array'] = np.full((width, height, 3), (0, 0, 0))
    return vr

def update_voxel_render(vr, player, width, height, draw_distance, current_time, world_scale):  # Added world_scale
    vr['ray_distance'] = int(draw_distance)
    sky_x_offset = int((player['angle'] / (2 * math.pi)) * width) % vr['sky_image'].get_width()
    screen_surface = pg.Surface((width, height))
    screen_surface.blit(vr['sky_image'], (-sky_x_offset, 0))
    screen_surface.blit(vr['sky_image'], (vr['sky_image'].get_width() - sky_x_offset, 0))
    vr['screen_array'] = pg.surfarray.array3d(screen_surface)
    breath_offset = BREATH_AMPLITUDE * math.sin(current_time * BREATH_FREQUENCY)
    effective_height = player['height'] + breath_offset
    vr['screen_array'] = ray_casting(
        vr['screen_array'], player['pos'], player['angle'], effective_height,
        player['pitch'], width, height, vr['delta_angle'], vr['ray_distance'],
        vr['h_fov'], vr['scale_height'], world_scale  # Pass world_scale
    )

def draw_voxel_render(vr, screen):
    pg.surfarray.blit_array(screen, vr['screen_array'])

# ------------------------------------------------------------
# Application Functions
# ------------------------------------------------------------
def init_app():
    app = {}
    app['width'], app['height'] = (800, 450)
    app['screen'] = pg.display.set_mode((app['width'], app['height']), pg.SCALED)
    app['clock'] = pg.time.Clock()
    app['draw_distance'] = DRAW_DISTANCE
    app['player'] = create_player()
    app['chunk_manager'] = init_chunk_manager()
    app['voxel_render'] = create_voxel_render(app['width'], app['height'])
    app['dt'] = 0.0
    app['world_scale'] = WORLD_SCALE  # Add world scale to app
    pg.mouse.set_visible(False)
    pg.event.set_grab(True)
    return app

def run_app():
    pg.init()
    app = init_app()
    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                pg.quit()
                return
        app['dt'] = app['clock'].tick(240) / 1000.0
        update_app(app)
        draw_app(app)
        pg.display.set_caption(
            f'FPS: {app["clock"].get_fps():.1f} | '
            f'X: {app["player"]["pos"][0]:.2f}, Y: {app["player"]["pos"][1]:.2f}, Z: {app["player"]["height"]:.2f} | '
            f'Rot X (Pitch): {app["player"]["pitch"]:.2f}, Rot Z (Yaw): {app["player"]["angle"]:.2f} | '
            f'Scale: {app["world_scale"]:.1f}'
        )

def update_app(app):
    mouse_rel = pg.mouse.get_rel()
    pressed_key = pg.key.get_pressed()
    
    if pressed_key[pg.K_PAGEUP]:
        app['draw_distance'] = min(3000, app['draw_distance'] + 50)
    if pressed_key[pg.K_PAGEDOWN]:
        app['draw_distance'] = max(500, app['draw_distance'] - 50)
    # Adjust world scale with + and -
    if pressed_key[pg.K_EQUALS] or pressed_key[pg.K_KP_PLUS]:  # + key
        app['world_scale'] = min(WORLD_SCALE_MAX, app['world_scale'] + WORLD_SCALE_STEP)
    if pressed_key[pg.K_MINUS] or pressed_key[pg.K_KP_MINUS]:  # - key
        app['world_scale'] = max(WORLD_SCALE_MIN, app['world_scale'] - WORLD_SCALE_STEP)

    current_time = pg.time.get_ticks()
    update_player(app['player'], mouse_rel, app['draw_distance'], app['chunk_manager'], app['dt'], app['world_scale'])
    update_voxel_render(app['voxel_render'], app['player'], app['width'], app['height'], app['draw_distance'], current_time, app['world_scale'])


def draw_app(app):
    draw_voxel_render(app['voxel_render'], app['screen'])
    pg.display.flip()


if __name__ == '__main__':
    run_app()