#version 450

layout(location = 0) in vec2 in_atlas_position;
layout(location = 1) in vec4 in_colour;

layout(location = 0) out vec4 out_colour;

layout(set = 0, binding = 1) uniform sampler2D glyph_atlas;

void main() {
  float alpha = in_colour.a * texture(glyph_atlas, in_atlas_position).a;
  out_colour = vec4(in_colour.rgb * alpha, alpha);
}
