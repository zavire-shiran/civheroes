#version 330

uniform mat4 WorldToCameraTransform;
uniform mat4 CameraToClipTransform;

layout(location=0) in vec2 position;
layout(location=1) in vec4 in_color;

smooth out vec4 color;

void main()
{
  vec4 pos = vec4(position.x, position.y, 0.0, 1.0);
  gl_Position = CameraToClipTransform * WorldToCameraTransform * pos;
  color = in_color;
}
