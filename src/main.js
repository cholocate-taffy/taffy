import * as THREE from 'three';
// GLTFLoader and other necessary modules
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { GPUComputationRenderer } from 'three/examples/jsm/misc/GPUComputationRenderer.js';
import { RGBELoader } from 'three/examples/jsm/loaders/RGBELoader.js';
import { SimplexNoise } from 'three/examples/jsm/math/SimplexNoise.js';

// =================================================================
// 初始化基础组件
// =================================================================
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(-80, 25, 50);
scene.add(camera);

const renderer = new THREE.WebGLRenderer({
  canvas: document.querySelector('#webgl-canvas'),
  antialias: true
});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 0.8;

const WIDTH = 128; 
const BOUNDS = 58; 
const BOUNDS_HALF = BOUNDS * 0.5; 

const heightmapFragmentShader = `
	#include <common>
	uniform vec2 mousePos;
	uniform float mouseSize;
	uniform float viscosity;
	uniform float deep;

	void main()	{
		vec2 cellSize = 1.0 / resolution.xy;
		vec2 uv = gl_FragCoord.xy * cellSize;
		// heightmapValue.x == 上一帧的高度
		// heightmapValue.y == 上上帧的高度
		vec4 heightmapValue = texture2D( heightmap, uv );
		vec4 north = texture2D( heightmap, uv + vec2( 0.0, cellSize.y ) );
		vec4 south = texture2D( heightmap, uv + vec2( 0.0, - cellSize.y ) );
		vec4 east = texture2D( heightmap, uv + vec2( cellSize.x, 0.0 ) );
		vec4 west = texture2D( heightmap, uv + vec2( - cellSize.x, 0.0 ) );
		float newHeight = ( ( north.x + south.x + east.x + west.x ) * 0.5 - heightmapValue.y ) * viscosity;
		// 鼠标影响
		float mousePhase = clamp( length( ( uv - vec2( 0.5 ) ) * BOUNDS - vec2( mousePos.x, -mousePos.y ) ) * PI / mouseSize, 0.0, PI );
		newHeight -= ( cos( mousePhase ) + 1.0 ) * deep;
		heightmapValue.y = heightmapValue.x;
		heightmapValue.x = newHeight;
		gl_FragColor = heightmapValue;
	}
`;

//读取水位和法线的着色器
const readWaterLevelFragmentShader = `
	uniform vec2 point1;
	uniform sampler2D levelTexture;

	// Integer to float conversion
	float shift_right( float v, float amt ) {
		v = floor( v ) + 0.5;
		return floor( v / exp2( amt ) );
	}
	float shift_left( float v, float amt ) {
		return floor( v * exp2( amt ) + 0.5 );
	}
	float mask_last( float v, float bits ) {
		return mod( v, shift_left( 1.0, bits ) );
	}
	float extract_bits( float num, float from, float to ) {
		from = floor( from + 0.5 ); to = floor( to + 0.5 );
		return mask_last( shift_right( num, from ), to - from );
	}
	vec4 encode_float( float val ) {
		if ( val == 0.0 ) return vec4( 0, 0, 0, 0 );
		float sign = val > 0.0 ? 0.0 : 1.0;
		val = abs( val );
		float exponent = floor( log2( val ) );
		float biased_exponent = exponent + 127.0;
		float fraction = ( ( val / exp2( exponent ) ) - 1.0 ) * 8388608.0;
		float t = biased_exponent / 2.0;
		float last_bit_of_biased_exponent = fract( t ) * 2.0;
		float remaining_bits_of_biased_exponent = floor( t );
		float byte4 = extract_bits( fraction, 0.0, 8.0 ) / 255.0;
		float byte3 = extract_bits( fraction, 8.0, 16.0 ) / 255.0;
		float byte2 = ( last_bit_of_biased_exponent * 128.0 + extract_bits( fraction, 16.0, 23.0 ) ) / 255.0;
		float byte1 = ( sign * 128.0 + remaining_bits_of_biased_exponent ) / 255.0;
		return vec4( byte4, byte3, byte2, byte1 );
	}

	void main()	{
		// 【修正】使用 WIDTH (输入纹理的宽度) 来计算正确的采样步长，而不是用 resolution (输出目标的尺寸)
		float aCellSize = 1.0 / WIDTH;
		float waterLevel = texture2D( levelTexture, point1 ).x;
		vec2 normal = vec2(
			( texture2D( levelTexture, point1 + vec2( -aCellSize, 0.0 ) ).x - texture2D( levelTexture, point1 + vec2( aCellSize, 0.0 ) ).x ) * WIDTH / BOUNDS,
			( texture2D( levelTexture, point1 + vec2( 0.0, -aCellSize ) ).x - texture2D( levelTexture, point1 + vec2( 0.0, aCellSize ) ).x ) * WIDTH / BOUNDS
		);
		if ( gl_FragCoord.x < 1.5 ) {
			gl_FragColor = encode_float( waterLevel );
		} else if ( gl_FragCoord.x < 2.5 ) {
			gl_FragColor = encode_float( normal.x );
		} else if ( gl_FragCoord.x < 3.5 ) {
			gl_FragColor = encode_float( normal.y );
		} else {
			gl_FragColor = encode_float( 0.0 );
		}
	}
`;


let waterMesh, meshRay, gpuCompute, heightmapVariable;
let duckModel, readWaterLevelShader, readWaterLevelRenderTarget, readWaterLevelImage;
const ducks = [];
const NUM_DUCKS = 7; 
let kirbyModel, animationMixer, kirbyActions = {};
let activeAction, previousAction;
const clock = new THREE.Clock();
const keyboardState = {};
const moveDirection = new THREE.Vector3();
const keyPressStartTime = {};
let joystickStartTime = 0;
let isMoving = false;
let layoutModel, layoutMixer, openAction, winAction, loseAction;
const downRaycaster = new THREE.Raycaster();
const forwardRaycaster = new THREE.Raycaster();
let kirbyVerticalVelocity = 0;
const gravity = -30;
let isGrounded = false;
const KIRBY_INITIAL_POSITION = new THREE.Vector3(-152.9, -14.4, -6.6);
const FALL_THRESHOLD = -150;
const interactiveButtons = [];
let doorButton, treeButton, waterButton;
let isDoorAnimationPlaying = false;
let isWinLoseAnimationPlaying = false;
const simplex = new SimplexNoise(); 
let frame = 0;

let isReady = false;
const cameraTargetPosition = new THREE.Vector3();
let isWaterMode = false;
const WATER_CAMERA_POSITION = new THREE.Vector3(57, 15, -115);
const WATER_CAMERA_LOOKAT = new THREE.Vector3(57, -19, -144);

const tmpQuat = new THREE.Quaternion();
const tmpQuatX = new THREE.Quaternion();
const tmpQuatZ = new THREE.Quaternion();
const yAxis = new THREE.Vector3(0, 1, 0);
const zAxis = new THREE.Vector3(0, 0, -1);

// =================================================================
// 舞台美术：环境、雾气、光照
// =================================================================
const rgbeLoader = new RGBELoader();
rgbeLoader.load('mysky.hdr', (texture) => {
  texture.mapping = THREE.EquirectangularReflectionMapping;
  scene.background = texture;
  scene.environment = texture;
});
scene.fog = new THREE.Fog(0x4a708e, 400, 600);
const ambientLight = new THREE.AmbientLight(0xffffff, 1.0);
scene.add(ambientLight);
const sunLight = new THREE.DirectionalLight(0xffffff, 1.9);
sunLight.position.set(5, 10.65, 7.5);
sunLight.castShadow = true;
sunLight.shadow.mapSize.set(2048, 2048);
scene.add(sunLight);

// =================================================================
// 水面和鸭子参数 
// =================================================================
const waterParams = {
  color: '#186a91',
  metalness: 0.9,
  roughness: 0.0,
  opacity: 0.67
};

function fillTexture(texture) {
  const waterMaxHeight = 0.1;

  function noise(x, y) {
    let multR = waterMaxHeight;
    let mult = 0.025;
    let r = 0;
    for (let i = 0; i < 15; i++) {
      r += multR * simplex.noise(x * mult, y * mult);
      multR *= 0.53 + 0.025 * i;
      mult *= 1.25;
    }
    return r;
  }

  const pixels = texture.image.data;
  let p = 0;
  for (let j = 0; j < WIDTH; j++) {
    for (let i = 0; i < WIDTH; i++) {
      const x = i * 128 / WIDTH;
      const y = j * 128 / WIDTH;
      pixels[p + 0] = noise(x, y);
      pixels[p + 1] = pixels[p + 0];
      pixels[p + 2] = 0;
      pixels[p + 3] = 1;
      p += 4;
    }
  }
}

function initWater() {
  const geometry = new THREE.PlaneGeometry(BOUNDS, BOUNDS, WIDTH - 1, WIDTH - 1);

  const material = new THREE.MeshStandardMaterial({
    color: new THREE.Color(waterParams.color),
    metalness: waterParams.metalness,
    roughness: waterParams.roughness,
    transparent: true,
    opacity: waterParams.opacity
  });

  material.onBeforeCompile = (shader) => {
    shader.uniforms.heightmap = { value: null };

    shader.vertexShader = 'uniform sampler2D heightmap;\n' + shader.vertexShader;
    shader.vertexShader = shader.vertexShader.replace(
      '#include <beginnormal_vertex>',
      `
            vec2 cellSize = vec2( 1.0 / ${WIDTH.toFixed(1)}, 1.0 / ${WIDTH.toFixed(1)} );
            vec3 objectNormal = vec3(
                ( texture2D( heightmap, uv + vec2( - cellSize.x, 0 ) ).x - texture2D( heightmap, uv + vec2( cellSize.x, 0 ) ).x ) * ${WIDTH.toFixed(1)} / ${BOUNDS.toFixed(1)},
                ( texture2D( heightmap, uv + vec2( 0, - cellSize.y ) ).x - texture2D( heightmap, uv + vec2( 0, cellSize.y ) ).x ) * ${WIDTH.toFixed(1)} / ${BOUNDS.toFixed(1)},
                1.0
            );
            `
    );
    shader.vertexShader = shader.vertexShader.replace(
      '#include <begin_vertex>',
      `
            float heightValue = texture2D( heightmap, uv ).x;
            vec3 transformed = vec3( position.x, position.y, heightValue * 6.0 ); // 乘以 6.0 来控制波浪高度
            `
    );
    material.userData.shader = shader;
  };

  waterMesh = new THREE.Mesh(geometry, material);
  waterMesh.rotation.x = -Math.PI / 2;
  waterMesh.position.set(57, -19, -144);
  waterMesh.receiveShadow = true;
  scene.add(waterMesh);

  meshRay = new THREE.Mesh(
    new THREE.PlaneGeometry(BOUNDS, BOUNDS, 1, 1),
    new THREE.MeshBasicMaterial({ color: 0xff0000, visible: false })
  );
  meshRay.rotation.x = -Math.PI / 2;
  meshRay.position.copy(waterMesh.position);
  scene.add(meshRay);

  gpuCompute = new GPUComputationRenderer(WIDTH, WIDTH, renderer);

  const heightmap0 = gpuCompute.createTexture();
  fillTexture(heightmap0);

  heightmapVariable = gpuCompute.addVariable('heightmap', heightmapFragmentShader, heightmap0);
  gpuCompute.setVariableDependencies(heightmapVariable, [heightmapVariable]);

  heightmapVariable.material.uniforms['mousePos'] = { value: new THREE.Vector2(10000, 10000) };
  heightmapVariable.material.uniforms['mouseSize'] = { value: 1.91 };
  heightmapVariable.material.uniforms['viscosity'] = { value: 0.98 };
  heightmapVariable.material.uniforms['deep'] = { value: 0.078 };
  heightmapVariable.material.defines.BOUNDS = BOUNDS.toFixed(1);

  const error = gpuCompute.init();
  if (error !== null) console.error(error);

  readWaterLevelShader = gpuCompute.createShaderMaterial(readWaterLevelFragmentShader, {
    point1: { value: new THREE.Vector2() },
    levelTexture: { value: null }
  });
  readWaterLevelShader.defines.WIDTH = WIDTH.toFixed(1);
  readWaterLevelShader.defines.BOUNDS = BOUNDS.toFixed(1);

  readWaterLevelImage = new Uint8Array(4 * 1 * 4);
  readWaterLevelRenderTarget = new THREE.WebGLRenderTarget(4, 1, {
    wrapS: THREE.ClampToEdgeWrapping,
    wrapT: THREE.ClampToEdgeWrapping,
    minFilter: THREE.NearestFilter,
    magFilter: THREE.NearestFilter,
    format: THREE.RGBAFormat,
    type: THREE.UnsignedByteType,
    depthBuffer: false
  });
}

initWater();

const controlParams = {
  speed: 30,
  jumpStrength: 15,
  rotationLerpFactor: 0.1,
  mouseSensitivity: 0.0023
};

const cameraParams = {
  distance: 72,
  height: 50,
  lag: 0.1,
  initialYaw: -1.58159,
  initialPitch: 0.114601
};

let cameraYaw = cameraParams.initialYaw;
let cameraPitch = cameraParams.initialPitch;
let targetCameraYaw = cameraYaw;
let targetCameraPitch = cameraParams.initialPitch;

const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
let isMouseDown = false;
let isDraggingOnWater = false;
let isOrbiting = false;

let isPinching = false;
let lastPinchDistance = 0;

// =================================================================
// 卡比移动控制
// =================================================================

function jump() {
  if (isGrounded) {
    kirbyVerticalVelocity = controlParams.jumpStrength;
    isGrounded = false;
  }
}

document.addEventListener('keydown', (event) => {
  const key = event.key.toLowerCase();
  if (['w', 'a', 's', 'd', 'arrowup', 'arrowdown', 'arrowleft', 'arrowright'].includes(key) && !keyPressStartTime[key]) {
    keyPressStartTime[key] = Date.now();
  }
  if (key === ' ') {
    jump();
  }
  keyboardState[key] = true;
});
document.addEventListener('keyup', (event) => {
  const key = event.key.toLowerCase();
  delete keyPressStartTime[key];
  keyboardState[key] = false;
});

const joystick = document.getElementById('joystick');
const joystickKnob = document.getElementById('joystick-knob');
const jumpButton = document.getElementById('jump-button');
let joystickActive = false;
let joystickStart = new THREE.Vector2();
let joystickCurrent = new THREE.Vector2();

joystick.addEventListener('touchstart', (event) => {
  event.preventDefault();
  joystickActive = true;
  joystickStartTime = Date.now();
  joystickStart.set(event.touches[0].clientX, event.touches[0].clientY);
}, { passive: false });

joystick.addEventListener('touchend', () => {
  joystickActive = false;
  joystickStartTime = 0;
  joystickKnob.style.transform = `translate(0px, 0px)`;
  keyboardState['w'] = false;
  keyboardState['s'] = false;
  keyboardState['a'] = false;
  keyboardState['d'] = false;
});

joystick.addEventListener('touchmove', (event) => {
  event.preventDefault();
  if (!joystickActive) return;
  joystickCurrent.set(event.touches[0].clientX, event.touches[0].clientY);
  const diff = joystickCurrent.clone().sub(joystickStart);
  const angle = diff.angle();
  const distance = Math.min(diff.length(), 50);
  const x = distance * Math.cos(angle);
  const y = distance * Math.sin(angle);
  joystickKnob.style.transform = `translate(${x}px, ${y}px)`;
  keyboardState['w'] = diff.y < -10;
  keyboardState['s'] = diff.y > 10;
  keyboardState['a'] = diff.x < -10;
  keyboardState['d'] = diff.x > 10;
}, { passive: false });

jumpButton.addEventListener('touchstart', (event) => {
  event.preventDefault();
  jump();
}, { passive: false });

function fadeToAction(name, duration) {
  if (!activeAction || !kirbyActions[name]) return;
  previousAction = activeAction;
  activeAction = kirbyActions[name];
  if (previousAction !== activeAction) {
    if (previousAction) previousAction.fadeOut(duration);
    activeAction
      .reset()
      .setEffectiveTimeScale(1)
      .setEffectiveWeight(1)
      .fadeIn(duration)
      .play();
  }
}

function updateKirbyMovement(deltaTime) {
  if (!kirbyModel || !animationMixer || !layoutModel) return;

  const speed = controlParams.speed * deltaTime;

  const cameraForward = new THREE.Vector3();
  camera.getWorldDirection(cameraForward);
  cameraForward.y = 0;
  cameraForward.normalize();

  const cameraRight = new THREE.Vector3();
  cameraRight.crossVectors(cameraForward, yAxis).normalize();

  moveDirection.set(0, 0, 0);

  if (keyboardState['w'] || keyboardState['arrowup']) moveDirection.add(cameraForward);
  if (keyboardState['s'] || keyboardState['arrowdown']) moveDirection.sub(cameraForward);
  if (keyboardState['a'] || keyboardState['arrowleft']) moveDirection.sub(cameraRight);
  if (keyboardState['d'] || keyboardState['arrowright']) moveDirection.add(cameraRight);

  isMoving = moveDirection.lengthSq() > 0.001;

  if (isMoving) {
    moveDirection.normalize();

    const targetAngle = Math.atan2(moveDirection.x, moveDirection.z);
    const targetQuaternion = new THREE.Quaternion();
    targetQuaternion.setFromAxisAngle(yAxis, targetAngle);
    kirbyModel.quaternion.slerp(targetQuaternion, controlParams.rotationLerpFactor);

    const origin = kirbyModel.position.clone().add(new THREE.Vector3(0, 2, 0));
    forwardRaycaster.set(origin, moveDirection);
    const wallIntersects = forwardRaycaster.intersectObject(layoutModel, true);
    let canMove = wallIntersects.length === 0 || wallIntersects[0].distance >= 1.5;

    if (canMove) {
      kirbyModel.position.add(moveDirection.multiplyScalar(speed));
    }
  }

  if (isMoving) {
    if (activeAction !== kirbyActions.run) fadeToAction('run', 0.2);
    let pressDuration = 0;
    if (joystickActive) {
      pressDuration = Date.now() - joystickStartTime;
    } else {
      const pressTimes = Object.values(keyPressStartTime);
      if (pressTimes.length > 0) {
        pressDuration = Date.now() - Math.min(...pressTimes);
      }
    }
    kirbyActions.run.setEffectiveTimeScale(pressDuration > 2000 ? 1.5 : 1.0);
  } else {
    if (activeAction !== kirbyActions.idle) fadeToAction('idle', 0.2);
  }

  const downOrigin = kirbyModel.position.clone().add(new THREE.Vector3(0, 1, 0));
  downRaycaster.set(downOrigin, new THREE.Vector3(0, -1, 0));
  const groundIntersects = downRaycaster.intersectObject(layoutModel, true);

  // 增大地面检测容错距离
  const onGround = groundIntersects.length > 0 && groundIntersects[0].distance < 1.5;

  if (onGround) {
    isGrounded = true;
    if (kirbyVerticalVelocity <= 0) {
      kirbyModel.position.y = groundIntersects[0].point.y;
      kirbyVerticalVelocity = 0;
    }
  } else {
    isGrounded = false;
  }

  if (!isGrounded) {
    kirbyVerticalVelocity += gravity * deltaTime;
  }

  kirbyModel.position.y += kirbyVerticalVelocity * deltaTime;

  if (kirbyModel.position.y < FALL_THRESHOLD) {
    kirbyModel.position.copy(KIRBY_INITIAL_POSITION);
    kirbyVerticalVelocity = 0;
  }

  animationMixer.update(deltaTime);
}

function updateCamera() {
  if (!kirbyModel) return;

  if (isWaterMode) {
    camera.position.lerp(WATER_CAMERA_POSITION, 0.05);
    cameraTargetPosition.lerp(WATER_CAMERA_LOOKAT, 0.05);
    camera.lookAt(cameraTargetPosition);
    return;
  }

  cameraYaw = THREE.MathUtils.lerp(cameraYaw, targetCameraYaw, 0.1);
  cameraPitch = THREE.MathUtils.lerp(cameraPitch, targetCameraPitch, 0.1);

  const pitchRotation = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), cameraPitch);
  const yawRotation = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), cameraYaw);
  const finalRotation = yawRotation.multiply(pitchRotation);

  const offset = new THREE.Vector3(0, cameraParams.height, cameraParams.distance);
  offset.applyQuaternion(finalRotation);

  const idealPosition = cameraTargetPosition.clone().add(offset);
  camera.position.lerp(idealPosition, cameraParams.lag);

  const lookAtPoint = cameraTargetPosition.clone().add(new THREE.Vector3(0, 5, 0));
  camera.lookAt(lookAtPoint);
}


// =================================================================
// 加载模型
// =================================================================
const gltfLoader = new GLTFLoader();

gltfLoader.load('布局.glb', (gltf) => {
  layoutModel = gltf.scene;
  layoutModel.position.set(0, 0, 0);
  layoutModel.scale.set(3.5, 3.5, 3.5);
  layoutModel.traverse((child) => {
    if (child.isMesh) {
      child.castShadow = true;
      child.receiveShadow = true;
    }
  });
  scene.add(layoutModel);

  layoutMixer = new THREE.AnimationMixer(layoutModel);
  const openClip = THREE.AnimationClip.findByName(gltf.animations, 'open');
  const winClip = THREE.AnimationClip.findByName(gltf.animations, 'win');
  const loseClip = THREE.AnimationClip.findByName(gltf.animations, 'lose');

  if (openClip) {
    openAction = layoutMixer.clipAction(openClip);
    openAction.setLoop(THREE.LoopOnce);
    openAction.clampWhenFinished = true;
  }
  if (winClip) {
    winAction = layoutMixer.clipAction(winClip);
    winAction.setLoop(THREE.LoopOnce);
    winAction.clampWhenFinished = true;
  }
  if (loseClip) {
    loseAction = layoutMixer.clipAction(loseClip);
    loseAction.setLoop(THREE.LoopOnce);
    loseAction.clampWhenFinished = true;
  }

  layoutMixer.addEventListener('finished', (e) => {
    if (e.action === openAction) isDoorAnimationPlaying = false;
    if (e.action === winAction || e.action === loseAction) {
      if (e.action.timeScale > 0) {
        e.action.paused = false;
        e.action.setLoop(THREE.LoopOnce);
        e.action.timeScale = -1;
        e.action.play();
      } else {
        e.action.stop();
        e.action.timeScale = 1;
        isWinLoseAnimationPlaying = false;
      }
    }
  });

  const buttonGeometry = new THREE.CylinderGeometry(1, 1, 0.5, 32);
  const buttonMaterial = new THREE.MeshStandardMaterial({ color: 0xffff00 });

  doorButton = new THREE.Mesh(buttonGeometry, buttonMaterial);
  doorButton.name = 'doorButton';
  doorButton.position.set(-79.4, -14, -7.4);
  doorButton.rotation.z = 1.63;
  doorButton.scale.set(1.5, 1.5, 1.5);
  scene.add(doorButton);
  interactiveButtons.push(doorButton);

  treeButton = new THREE.Mesh(buttonGeometry, buttonMaterial);
  treeButton.name = 'treeButton';
  treeButton.position.set(20, -25, 95);
  treeButton.scale.set(4, 4, 4);
  scene.add(treeButton);
  interactiveButtons.push(treeButton);

  waterButton = new THREE.Mesh(buttonGeometry, buttonMaterial);
  waterButton.name = 'waterButton';
  waterButton.position.set(84, -14.5, -120);
  waterButton.rotation.set(Math.PI, Math.PI, 1.74);
  waterButton.scale.set(2.5, 2.5, 2.5);
  scene.add(waterButton);
  interactiveButtons.push(waterButton);

  if (kirbyModel) isReady = true;
});

gltfLoader.load('kirby.glb', (gltf) => {
  kirbyModel = gltf.scene;
  kirbyModel.position.copy(KIRBY_INITIAL_POSITION);
  kirbyModel.rotation.y = 4.91;
  kirbyModel.scale.set(2.5, 2.5, 2.5);
  kirbyModel.traverse((child) => {
    if (child.isMesh) child.castShadow = true;
  });
  scene.add(kirbyModel);

  cameraTargetPosition.copy(kirbyModel.position);

  animationMixer = new THREE.AnimationMixer(kirbyModel);
  const idleClip = THREE.AnimationClip.findByName(gltf.animations, 'idle');
  const runClip = THREE.AnimationClip.findByName(gltf.animations, 'run');

  if (idleClip) {
    kirbyActions.idle = animationMixer.clipAction(idleClip);
    activeAction = kirbyActions.idle;
    activeAction.play();
  }
  if (runClip) kirbyActions.run = animationMixer.clipAction(runClip);

  if (layoutModel) isReady = true;
});

gltfLoader.load('Duck.glb', (gltf) => {
  duckModel = gltf.scene;
  duckModel.scale.set(0.022, 0.022, 0.022); 
  duckModel.traverse(child => {
    if (child.isMesh) {
      child.castShadow = true;
    }
  });
  createDucks();
});

function createDucks() {
  for (let i = 0; i < NUM_DUCKS; i++) {
    const duck = duckModel.clone();
    duck.position.x = (Math.random() - 0.5) * BOUNDS * 0.7;
    duck.position.z = (Math.random() - 0.5) * BOUNDS * 0.7;
    duck.position.add(waterMesh.position);
    duck.userData.velocity = new THREE.Vector3();
    scene.add(duck);
    ducks.push(duck);
  }
}

// duckDynamics 函数
function duckDynamics() {
  const heightmapTexture = gpuCompute.getCurrentRenderTarget(heightmapVariable).texture;
  readWaterLevelShader.uniforms['levelTexture'].value = heightmapTexture;

  for (let i = 0; i < NUM_DUCKS; i++) {
    const sphere = ducks[i];

    if (sphere) {
      const u = (sphere.position.x - waterMesh.position.x) / BOUNDS + 0.5;
      const v = 1.0 - ((sphere.position.z - waterMesh.position.z) / BOUNDS + 0.5);
      readWaterLevelShader.uniforms['point1'].value.set(u, v);

      gpuCompute.doRenderTarget(readWaterLevelShader, readWaterLevelRenderTarget);

      renderer.readRenderTargetPixels(readWaterLevelRenderTarget, 0, 0, 4, 1, readWaterLevelImage);
      const pixels = new Float32Array(readWaterLevelImage.buffer);

      const waterNormal = new THREE.Vector3(pixels[1], 0, -pixels[2]);

      const pos = sphere.position;
      const startPos = pos.clone();

      pos.y = waterMesh.position.y + pixels[0] * 6.0;

      waterNormal.multiplyScalar(0.3);
      sphere.userData.velocity.add(waterNormal);

      const drift = new THREE.Vector3(
        (Math.sin(clock.elapsedTime * 0.5 + i * 2.1) * 0.001),
        0,
        (Math.cos(clock.elapsedTime * 0.5 + i * 3.5) * 0.001)
      );
      sphere.userData.velocity.add(drift);

      sphere.userData.velocity.multiplyScalar(0.995);
      pos.add(sphere.userData.velocity);

      const decal = 0.001;
      const limit = BOUNDS_HALF - 2;
      const currentPos = pos.clone().sub(waterMesh.position);

      if (currentPos.x < -limit) {
        pos.x = waterMesh.position.x - limit + decal;
        sphere.userData.velocity.x *= -0.54;
      } else if (currentPos.x > limit) {
        pos.x = waterMesh.position.x + limit - decal;
        sphere.userData.velocity.x *= -0.54;
      }

      if (currentPos.z < -limit) {
        pos.z = waterMesh.position.z - limit + decal;
        sphere.userData.velocity.z *= -0.54;
      } else if (currentPos.z > limit) {
        pos.z = waterMesh.position.z + limit - decal;
        sphere.userData.velocity.z *= -0.54;
      }

      const surfaceNormal = new THREE.Vector3(pixels[1], 1, -pixels[2]).normalize();
      const moveDirection = pos.clone().sub(startPos);
      moveDirection.y = 0;
      moveDirection.normalize();

      if (moveDirection.lengthSq() > 0.0001) {
        tmpQuatX.setFromUnitVectors(zAxis, moveDirection);
      }

      tmpQuatZ.setFromUnitVectors(yAxis, surfaceNormal);
      tmpQuat.multiplyQuaternions(tmpQuatZ, tmpQuatX);
      sphere.quaternion.slerp(tmpQuat, 0.017);
    }
  }
}

// =================================================================
// 记忆星星游戏
// =================================================================

const gameOverlay = document.getElementById('game-overlay');
const closeGameButton = document.getElementById('close-game-button');
const startGameButton = document.getElementById('start-game-button');
const gameStatus = document.getElementById('game-status');
const gameBoard = document.getElementById('memory-game-board');
const stars = document.querySelectorAll('.memory-star');

const memoryGame = {
  level: 0,
  sequence: [],
  playerSequence: [],
  state: 'idle',
  colors: ['green', 'red', 'yellow', 'blue', 'orange', 'purple'],
  WIN_LEVEL: 4,
  LIGHT_UP_DURATION: 350,
  LIGHT_UP_DELAY: 150,
  lastResult: null,
};

const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

function initGameUI() {
  gameStatus.innerHTML = '记住闪烁的星星顺序并重复它。';
  gameStatus.className = '';
  startGameButton.style.display = 'inline-block';
  closeGameButton.style.display = 'none';
  gameBoard.classList.remove('player-turn');
  memoryGame.lastResult = null;
}

function startGame() {
  memoryGame.level = 0;
  memoryGame.sequence = [];
  memoryGame.state = 'computer-turn';
  startGameButton.style.display = 'none';
  closeGameButton.style.display = 'inline-block';
  nextLevel();
}

async function nextLevel() {
  memoryGame.level++;
  memoryGame.playerSequence = [];
  memoryGame.state = 'computer-turn';
  gameStatus.textContent = `第 ${memoryGame.level} 关`;
  gameStatus.className = '';
  gameBoard.classList.remove('player-turn');

  let stepsToAdd = (memoryGame.level === 1) ? 1 : (memoryGame.level === memoryGame.WIN_LEVEL) ? 3 : 2;

  for (let i = 0; i < stepsToAdd; i++) {
    const randomIndex = Math.floor(Math.random() * memoryGame.colors.length);
    memoryGame.sequence.push(memoryGame.colors[randomIndex]);
  }

  await sleep(1000);
  await playSequence();
}

async function playSequence() {
  const duration = Math.max(150, memoryGame.LIGHT_UP_DURATION - (memoryGame.level * 25));
  const delay = Math.max(100, memoryGame.LIGHT_UP_DELAY - (memoryGame.level * 15));

  for (const color of memoryGame.sequence) {
    await lightUpStar(color, duration);
    await sleep(delay);
  }

  memoryGame.state = 'player-turn';
  gameStatus.textContent = '到你了！';
  gameBoard.classList.add('player-turn');
}

function lightUpStar(color, duration = memoryGame.LIGHT_UP_DURATION) {
  return new Promise(resolve => {
    const star = document.getElementById(`star-${color}`);
    star.classList.add('lit');
    setTimeout(() => {
      star.classList.remove('lit');
      resolve();
    }, duration);
  });
}

function handlePlayerInput(color) {
  if (memoryGame.state !== 'player-turn') return;
  lightUpStar(color);
  memoryGame.playerSequence.push(color);
  const currentStep = memoryGame.playerSequence.length - 1;

  if (memoryGame.playerSequence[currentStep] !== memoryGame.sequence[currentStep]) {
    endGame(false);
    return;
  }

  if (memoryGame.playerSequence.length === memoryGame.sequence.length) {
    if (memoryGame.level >= memoryGame.WIN_LEVEL) {
      endGame(true);
    } else {
      memoryGame.state = 'computer-turn';
      gameBoard.classList.remove('player-turn');
      setTimeout(nextLevel, 1200);
    }
  }
}

function endGame(didWin) {
  memoryGame.state = 'game-over';
  gameBoard.classList.remove('player-turn');
  startGameButton.style.display = 'none';

  if (didWin) {
    memoryGame.lastResult = 'win';
    gameStatus.innerHTML = "恭喜你帮卡比寻找到了记忆star，<br>请返回查看小彩蛋吧";
    gameStatus.className = 'win';
  } else {
    memoryGame.lastResult = 'lose';
    gameStatus.innerHTML = "很遗憾没有成功，<br>请返回树下查看星星雨吧";
    gameStatus.className = 'lose';
    gameBoard.classList.add('shake');
  }
}

startGameButton.addEventListener('click', startGame);

closeGameButton.addEventListener('click', () => {
  gameOverlay.classList.remove('visible');
  setTimeout(() => {
    gameOverlay.style.display = 'none';
    if (memoryGame.lastResult === 'win' && winAction && !isWinLoseAnimationPlaying) {
      isWinLoseAnimationPlaying = true;
      winAction.reset().setEffectiveTimeScale(0.8).play();
    } else if (memoryGame.lastResult === 'lose' && loseAction && !isWinLoseAnimationPlaying) {
      isWinLoseAnimationPlaying = true;
      loseAction.reset().setEffectiveTimeScale(0.8).play();
    }
    memoryGame.state = 'idle';
    initGameUI();
  }, 500);
});

stars.forEach(star => {
  star.addEventListener('click', (e) => handlePlayerInput(e.currentTarget.dataset.color));
});

gameBoard.addEventListener('animationend', () => gameBoard.classList.remove('shake'));

// =================================================================
// 音乐控制逻辑
// =================================================================
const musicControl = document.getElementById('music-control');
const bgMusic = document.getElementById('bg-music');
const iconMusicOn = document.getElementById('icon-music-on');
const iconMusicOff = document.getElementById('icon-music-off');

const unlockAudio = () => {
  bgMusic.play().then(() => bgMusic.pause()).catch(e => { });
};
document.body.addEventListener('pointerdown', unlockAudio, { once: true });

musicControl.addEventListener('click', () => {
  if (bgMusic.paused) {
    bgMusic.play().catch(e => console.error("音乐播放失败:", e));
  } else {
    bgMusic.pause();
  }
});

bgMusic.onplay = () => {
  iconMusicOn.style.display = 'block';
  iconMusicOff.style.display = 'none';
};

bgMusic.onpause = () => {
  iconMusicOn.style.display = 'none';
  iconMusicOff.style.display = 'block';
};

// =================================================================
// 主交互逻辑
// =================================================================
document.addEventListener('pointerdown', (event) => {
  if (event.target.closest('#music-control')) return; // GUI check removed
  if (gameOverlay.classList.contains('visible') && !event.target.closest('#game-content')) return;

  isMouseDown = true;
  mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
  raycaster.setFromCamera(mouse, camera);

  const waterIntersects = raycaster.intersectObject(meshRay);
  if (waterIntersects.length > 0) {
    isDraggingOnWater = true;
    const point = waterIntersects[0].point;
    const localX = point.x - waterMesh.position.x;
    const localZ = point.z - waterMesh.position.z;
    heightmapVariable.material.uniforms.mousePos.value.set(localX, localZ);
    return;
  }

  const buttonIntersects = raycaster.intersectObjects(interactiveButtons);
  if (buttonIntersects.length > 0) {
    const clickedButton = buttonIntersects[0].object;
    if (clickedButton.name === 'doorButton' && openAction && !isDoorAnimationPlaying) {
      isDoorAnimationPlaying = true;
      openAction.reset().play();
    } else if (clickedButton.name === 'treeButton') {
      gameOverlay.style.display = 'flex';
      setTimeout(() => gameOverlay.classList.add('visible'), 10);
      initGameUI();
    } else if (clickedButton.name === 'waterButton') {
      isWaterMode = !isWaterMode;
      kirbyModel.visible = !isWaterMode;
      joystick.style.display = isWaterMode ? 'none' : 'flex';
      jumpButton.style.display = isWaterMode ? 'none' : 'block';
      if (!isWaterMode) {
        cameraTargetPosition.copy(kirbyModel.position);
      }
    }
  } else {
    if (!isPinching) {
      isOrbiting = true;
    }
  }
});

document.addEventListener('pointerup', () => {
  isMouseDown = false;
  isDraggingOnWater = false;
  isOrbiting = false;
  if (heightmapVariable) {
    heightmapVariable.material.uniforms.mousePos.value.set(10000, 10000);
  }
});

document.addEventListener('pointermove', (event) => {
  if (isOrbiting && !isWaterMode) {
    targetCameraYaw -= event.movementX * controlParams.mouseSensitivity;
    targetCameraPitch -= event.movementY * controlParams.mouseSensitivity;
    targetCameraPitch = Math.max(-Math.PI / 4, Math.min(Math.PI / 2, targetCameraPitch));
  }

  if (isMouseDown && isDraggingOnWater) {
    mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObject(meshRay);
    if (intersects.length > 0) {
      const point = intersects[0].point;
      const localX = point.x - waterMesh.position.x;
      const localZ = point.z - waterMesh.position.z;
      heightmapVariable.material.uniforms.mousePos.value.set(localX, localZ);
    }
  }
});

document.addEventListener('wheel', (event) => {
  cameraParams.distance += event.deltaY * 0.05;
  cameraParams.distance = Math.max(15, Math.min(80, cameraParams.distance));
});

document.addEventListener('touchstart', (event) => {
  if (event.touches.length === 2) {
    isPinching = true;
    isOrbiting = false;
    const dx = event.touches[0].clientX - event.touches[1].clientX;
    const dy = event.touches[0].clientY - event.touches[1].clientY;
    lastPinchDistance = Math.sqrt(dx * dx + dy * dy);
  }
}, { passive: false });

document.addEventListener('touchmove', (event) => {
  if (isPinching && event.touches.length === 2) {
    event.preventDefault();
    const dx = event.touches[0].clientX - event.touches[1].clientX;
    const dy = event.touches[0].clientY - event.touches[1].clientY;
    const currentPinchDistance = Math.sqrt(dx * dx + dy * dy);
    const deltaDistance = lastPinchDistance - currentPinchDistance;

    cameraParams.distance += deltaDistance * 0.25;
    cameraParams.distance = Math.max(15, Math.min(80, cameraParams.distance));

    lastPinchDistance = currentPinchDistance;
  }
}, { passive: false });

document.addEventListener('touchend', (event) => {
  if (event.touches.length < 2) {
    isPinching = false;
  }
});

function animate() {
  requestAnimationFrame(animate);
  const deltaTime = clock.getDelta();

  // GPGPU 计算和鸭子动态
  frame++;
  if (frame >= 7 - 4) {
    if (gpuCompute && waterMesh && waterMesh.material.userData.shader) {
      gpuCompute.compute();
      const heightmapTexture = gpuCompute.getCurrentRenderTarget(heightmapVariable).texture;
      waterMesh.material.userData.shader.uniforms.heightmap.value = heightmapTexture;
      if (ducks.length > 0) { 
        duckDynamics();
      }
    }
    frame = 0;
  }

  if (layoutMixer) {
    layoutMixer.update(deltaTime);
  }

  if (isReady) {
    if (kirbyModel && !isWaterMode) {
      cameraTargetPosition.lerp(kirbyModel.position, 0.08);
    }
    updateKirbyMovement(deltaTime);
    updateCamera();
  }

  renderer.render(scene, camera);
}
animate();

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

