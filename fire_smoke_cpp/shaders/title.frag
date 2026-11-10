#version 330 core
in vec2 v_uv;
out vec4 fragColor;
uniform sampler2D title_tex;
void main() {
    fragColor = texture(title_tex, v_uv);
}
