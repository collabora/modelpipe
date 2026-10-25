#version 450

// matches GLYPH_ATLAS_SIZE in overlay_vulkan.py
const float GLYPH_ATLAS_SIZE = 1024.0;

// the quad's fixed vertex format has no colour slot, z and uv carry packed integers
layout(location = 0) in vec3 in_position_and_texel;
layout(location = 1) in vec2 in_packed_colour;

layout(location = 0) out vec2 out_atlas_position;
layout(location = 1) out vec4 out_colour;

void main() {
  gl_Position = vec4(in_position_and_texel.xy, 0.0, 1.0);
  float texel = in_position_and_texel.z;
  vec2 atlas_texel = vec2(mod(texel, GLYPH_ATLAS_SIZE), floor(texel / GLYPH_ATLAS_SIZE));
  out_atlas_position = atlas_texel / GLYPH_ATLAS_SIZE;
  vec4 red_blue_green_alpha = vec4(floor(in_packed_colour / 256.0), mod(in_packed_colour, 256.0));
  out_colour = red_blue_green_alpha.xzyw / 255.0;
}
