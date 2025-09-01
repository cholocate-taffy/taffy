uniform float waterSize;
uniform float time;
    
varying vec3 vNormal;
varying vec3 vPosition;

void main() {
    vec3 newPosition = position;
    
    // 基于时间和位置生成波浪效果
    float waveHeight = sin(newPosition.x * 0.1 + time) * cos(newPosition.z * 0.1 + time) * 10.0;
    newPosition.y = waveHeight;
        
    // 传递法线和位置到片元着色器
    vNormal = normal;
    vPosition = newPosition;
        
    gl_Position = projectionMatrix * modelViewMatrix * vec4( newPosition, 1.0 );
}