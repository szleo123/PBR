#version 330 core

// Compute the irradiance within the glossy
// BRDF lobe aligned with a hard-coded wi
// that will equal our surface normal direction.
// Our surface normal direction is normalize(fs_Pos).

in vec3 fs_Pos;
out vec4 out_Col;
uniform samplerCube u_EnvironmentMap;
uniform float u_Roughness;

const float PI = 3.14159265359;
const uint SAMPLE_COUNT = 1024u;

float RadicalInverse_VdC(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}

vec2 Hammersley(uint i, uint N) {
    return vec2(float(i)/float(N), RadicalInverse_VdC(i));
}

vec3 ImportanceSampleGGX(vec2 xi, vec3 N, float roughness) {
    float a = roughness * roughness;
    float phi = 2.0 * PI * xi.x;
    float cosTheta = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
    float sinTheta = sqrt(1.0 - cosTheta*cosTheta);
    // from spherical coordinates to cartesian coordinates - halfway vector
    vec3 wh;
    wh.x = cos(phi) * sinTheta;
    wh.y = sin(phi) * sinTheta;
    wh.z = cosTheta;
    // from tangent-space H vector to world-space sample vector
    vec3 up        = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent   = normalize(cross(up, N));
    vec3 bitangent = cross(N, tangent);

    vec3 whW = tangent * wh.x + bitangent * wh.y + N * wh.z;
    return normalize(whW);
}

float DistributionGGX(vec3 wo, vec3 wh, float roughness) {
    float alpha = roughness * roughness;
    float b = dot(wo, wh);
    return alpha*alpha / pow(b * b * (alpha*alpha - 1) + 1, 2) / PI;
}

void main() {
    // TODO
    vec3 wo = normalize(fs_Pos);
    const uint SAMPLE_COUNT = 1024u;
    float totalWeight = 0.0;
    vec3 prefilteredColor = vec3(0.0);
    for(uint i = 0u; i < SAMPLE_COUNT; ++i) {
        vec2 xi = Hammersley(i, SAMPLE_COUNT);
        vec3 wh = ImportanceSampleGGX(xi, wo, u_Roughness);
        // Reflect wo about wh
        vec3 wi  = normalize(2.0 * dot(wo, wh) * wh - wo);
        float NdotL = max(dot(wo, wi), 0.0);
        if(NdotL > 0.0){
//            prefilteredColor += texture(u_EnvironmentMap, wi).rgb * NdotL;
//            totalWeight      += NdotL;
            float D       = DistributionGGX(wo, wh, u_Roughness);
            float nDotwh  = max(dot(wo, wh), 0.0);
            float woDotwh = max(dot(wh, wo), 0.0);
            float pdf = D * nDotwh / (4.0 * woDotwh) + 0.0001;
            float resolution = 1024.0;
            float saTexel  = 4.0 * PI / (6.0 * resolution * resolution);
            float saSample = 1.0 / (float(SAMPLE_COUNT) * pdf + 0.0001);
            float mipLevel = u_Roughness == 0.0 ? 0.0 : 0.5 * log2(saSample / saTexel);
            prefilteredColor += textureLod(u_EnvironmentMap, wi, mipLevel).rgb * NdotL;
            totalWeight += NdotL;
        }
    }
    prefilteredColor = prefilteredColor / totalWeight;
    out_Col = vec4(prefilteredColor, 1.f);
}
