uniform vec3 lightPosition;
uniform vec3 lightColor;
uniform vec3 waterColor;
uniform float shininess;
uniform float reflectivity;
uniform samplerCube tCube;

varying vec3 vNormal;
varying vec3 vPosition;

void main() {
    vec3 normal = vNormal;
    vec3 lightDirection = normalize(lightPosition - vPosition);
    vec3 viewDirection = normalize(-vPosition);
    vec3 reflectDirection = reflect(-lightDirection, normal);

    // 环境光
    vec3 ambient = lightColor * 0.2;

    // 漫反射
    float diff = max(dot(normal, lightDirection), 0.0);
    vec3 diffuse = lightColor * diff;

    // 高光
    vec3 specular = vec3(0.0);
    float spec = pow(max(dot(reflectDirection, viewDirection), 0.0), shininess);
    specular = lightColor * spec;

    // 反射
    vec3 reflection = textureCube(tCube, reflectDirection).rgb;
    vec3 totalReflection = reflection * reflectivity;

    // 最终颜色
    vec3 finalColor = waterColor * (ambient + diffuse) + specular + totalReflection;
    gl_FragColor = vec4(finalColor, 1.0);
}