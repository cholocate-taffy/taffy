import * as THREE from 'three';

// 1. 基本设置
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111111); // 设置一个深灰色背景，而不是纯黑

const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({ antialias: true });

renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

// 2. 添加一个最简单的、会发光的旋转立方体
const geometry = new THREE.BoxGeometry(1, 1, 1);
// 使用不需要光照的 MeshBasicMaterial
const material = new THREE.MeshBasicMaterial({ color: 0x00ff00, wireframe: true });
const cube = new THREE.Mesh(geometry, material);
scene.add(cube);

// 3. 设置相机位置
camera.position.z = 5;

// 4. 创建一个动画循环来渲染场景
function animate() {
  requestAnimationFrame(animate);

  // 让立方体旋转，这样我们能确定渲染循环在工作
  cube.rotation.x += 0.01;
  cube.rotation.y += 0.01;

  renderer.render(scene, camera);
}

// 5. 处理窗口大小变化
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// 启动动画
animate();

console.log('简化的测试脚本已运行');

