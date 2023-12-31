#version 330 core
uniform vec3 u_CamPos;

// PBR material attributes
uniform vec3 u_Albedo;
uniform float u_Metallic;
uniform float u_Roughness;
uniform float u_AmbientOcclusion;
// Texture maps for controlling some of the attribs above, plus normal mapping
uniform sampler2D u_AlbedoMap;
uniform sampler2D u_MetallicMap;
uniform sampler2D u_RoughnessMap;
uniform sampler2D u_AOMap;
uniform sampler2D u_NormalMap;
// If true, use the textures listed above instead of the GUI slider values
uniform bool u_UseAlbedoMap;
uniform bool u_UseMetallicMap;
uniform bool u_UseRoughnessMap;
uniform bool u_UseAOMap;
uniform bool u_UseNormalMap;

// Image-based lighting
uniform samplerCube u_DiffuseIrradianceMap;
uniform samplerCube u_GlossyIrradianceMap;
uniform sampler2D u_BRDFLookupTexture;

// Varyings
in vec3 fs_Pos;
in vec3 fs_Nor; // Surface normal
in vec3 fs_Tan; // Surface tangent
in vec3 fs_Bit; // Surface bitangent
in vec2 fs_UV;
out vec4 out_Col;

const float PI = 3.14159f;


// Set the input material attributes to texture-sampled values
// if the indicated booleans are TRUE
void handleMaterialMaps(inout vec3 albedo, inout float metallic,
                        inout float roughness, inout float ambientOcclusion,
                        inout vec3 normal) {
    if(u_UseAlbedoMap) {
        albedo = pow(texture(u_AlbedoMap, fs_UV).rgb, vec3(2.2));
    }
    if(u_UseMetallicMap) {
        metallic = texture(u_MetallicMap, fs_UV).r;
    }
    if(u_UseRoughnessMap) {
        roughness = texture(u_RoughnessMap, fs_UV).r;
    }
    if(u_UseAOMap) {
        ambientOcclusion = texture(u_AOMap, fs_UV).r;
    }
    if(u_UseNormalMap) {
        // TODO: Apply normal mapping
       vec3 newNor = normalize(texture(u_NormalMap, fs_UV).rgb);
       normal = mat3(fs_Tan, fs_Bit, normal) * newNor;
    }
}

vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness)
{
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}


void main()
{
    vec3  N                = fs_Nor;
    vec3  albedo           = u_Albedo;
    float metallic         = u_Metallic;
    float roughness        = u_Roughness;
    float ambientOcclusion = u_AmbientOcclusion;

    handleMaterialMaps(albedo, metallic, roughness, ambientOcclusion, N);

    vec3 wo = normalize(u_CamPos - fs_Pos);
    vec3 wi = reflect(-wo, N);
    // diffuse part
    vec3 R = albedo;
    vec3 diffuseIrradiance = texture(u_DiffuseIrradianceMap, N).rgb;

    // specular part
    vec3 plastic_CT_color = vec3(0.04);
    vec3 metallic_CT_color = albedo;
    vec3 fresnel_color = mix(plastic_CT_color, metallic_CT_color, metallic);
    vec3 F = fresnelSchlickRoughness(max(dot(N, wo), 0.0), fresnel_color, roughness);
    vec3 kS = F;
    vec3 kD = vec3(1.0f) - kS;
    kD *= 1.0 - metallic;
    const float MAX_REFLECTION_LOD = 4.0;
    vec3 prefilteredColor = textureLod(u_GlossyIrradianceMap, wi,  roughness * MAX_REFLECTION_LOD).rgb;
    vec2 envBRDF  = texture(u_BRDFLookupTexture, vec2(max(dot(N, wo), 0.0), roughness)).rg;
    vec3 specularIrradiance = prefilteredColor * (F * envBRDF.x + envBRDF.y);


    vec3 result = kD * R * diffuseIrradiance + specularIrradiance;
    result *= ambientOcclusion;
    result = result / (vec3(1.f) + result);
    result = pow(result, vec3(1.f / 2.2f));
    out_Col = vec4(result, 1.f);
}
