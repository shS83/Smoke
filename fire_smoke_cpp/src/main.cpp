#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include <GL/glew.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int W = 240, H = 135, N = W * H;
constexpr float pi = 3.14159265358979323846f;
constexpr int pressureIterations = 30;
constexpr float maxFuel = 2.0f, maxTemp = 3.2f;
float clamp(float v, float low, float high) { return std::clamp(v, low, high); }
int ix(int x, int y) { return y * W + x; }

struct Field {
    std::vector<float> v;
    Field() : v(N, 0.0f) {}
    float &at(int x, int y) { return v[ix(x, y)]; }
    float at(int x, int y) const { return v[ix(x, y)]; }
    void clear() { std::fill(v.begin(), v.end(), 0.0f); }
};

float bilinear(const Field &f, float x, float y) {
    x = clamp(x, 0.f, W - 1.001f);
    y = clamp(y, 0.f, H - 1.001f);
    const int x0 = static_cast<int>(x), y0 = static_cast<int>(y);
    const int x1 = std::min(x0 + 1, W - 1), y1 = std::min(y0 + 1, H - 1);
    const float sx = x - static_cast<float>(x0), sy = y - static_cast<float>(y0);
    return (1.f - sy) * ((1.f - sx) * f.at(x0, y0) + sx * f.at(x1, y0))
         + sy * ((1.f - sx) * f.at(x0, y1) + sx * f.at(x1, y1));
}

void advect(Field &target, const Field &source, const Field &vx, const Field &vy, float dt) {
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            int k = ix(x, y);
            target.v[k] = bilinear(source, x - vx.v[k] * dt, y - vy.v[k] * dt);
        }
}

float blob(float x, float y, float cx, float cy, float rx, float ry) {
    float dx = (x - cx) / rx, dy = (y - cy) / ry;
    return std::exp(-2.f * (dx * dx + dy * dy));
}

struct Simulation {
    Field density, temp, fuel, burn, vx, vy, divergence, pressure, tmpPressure;
    Field curl, gradX, gradY, oldVx, oldVy, tempField;
    float elapsed = 0.f, divergenceRms = 0.f, curlRms = 0.f;
    float intensity = 1.f;
    std::vector<float> obstacle = std::vector<float>(N, 0.f);
    bool obstacleEnabled = true;
    bool showFire = true;
    int debugMode = 0;

    void reset() {
        for (Field *f : {&density, &temp, &fuel, &burn, &vx, &vy, &divergence,
                         &pressure, &tmpPressure, &curl, &gradX, &gradY, &oldVx, &oldVy}) f->clear();
        elapsed = divergenceRms = curlRms = 0.f;
    }

    void addSource(float dt) {
        const float pulse = 0.90f + 0.10f * std::sin(elapsed * 3.2f);
        const float sway = std::sin(elapsed * .91f) * 2.2f + std::sin(elapsed * .37f + 1.3f);
        for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
            const int k = ix(x, y);
            // v1.2: triple the v1.1 flame source width, one connected fire.
            const float s = blob(float(x), float(y), W * .5f, H - 7.f, 38.4f, 4.4f) * pulse * intensity;
            fuel.v[k] = std::min(maxFuel, fuel.v[k] + s * 5.0f * dt);
            temp.v[k] = std::min(maxTemp, temp.v[k] + s * 6.6f * dt);
            density.v[k] += s * 0.82f * dt;
            vy.v[k] -= s * 14.0f * dt;
            vx.v[k] += s * sway * dt * 2.8f;
        }
    }

    void addMouse(float mx, float my, int ww, int wh, float dt) {
        if (ww <= 0 || wh <= 0) return;
        const float sx = mx / ww * W, sy = my / wh * H;
        for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
            int k = ix(x, y);
            float s = blob(float(x), float(y), sx, sy, 5.f, 5.f);
            fuel.v[k] = std::min(maxFuel, fuel.v[k] + s * 5.f * dt);
            temp.v[k] = std::min(maxTemp, temp.v[k] + s * 6.f * dt);
            density.v[k] += s * 0.65f * dt;
            vy.v[k] -= s * 9.f * dt;
        }
    }

    void combust(float dt) {
        for (int k = 0; k < N; ++k) {
            float ignition = clamp((temp.v[k] - 0.32f) / (1.f - 0.32f), 0.f, 1.f);
            float burned = std::min(fuel.v[k], fuel.v[k] * ignition * 1.85f * dt);
            fuel.v[k] -= burned;
            temp.v[k] = std::min(maxTemp, temp.v[k] + burned * 2.45f);
            density.v[k] += burned * 0.92f; // v0.8.2: 0.38
            burn.v[k] = dt > 0 ? burned / dt : 0.f;
        }
    }

    void velocityBoundaries() {
        for (int y = 0; y < H; ++y) {
            vx.at(0,y) = vx.at(W-1,y) = 0.f;
            vy.at(0,y) = vy.at(1,y);
            vy.at(W-1,y) = vy.at(W-2,y);
        }
        for (int x = 0; x < W; ++x) {
            vy.at(x,H-1) = 0.f;
            vx.at(x,H-1) = vx.at(x,H-2);
            vx.at(x,0) = vx.at(x,1);
            vy.at(x,0) = std::min(vy.at(x,1), 0.f);
        }
        if (obstacleEnabled) for (int k = 0; k < N; ++k) if (obstacle[k] > .30f) {
            vx.v[k] = vy.v[k] = 0.f;
        }
    }

    static void pressureBoundaries(Field &p) {
        for (int y = 0; y < H; ++y) { p.at(0,y) = p.at(1,y); p.at(W-1,y) = p.at(W-2,y); }
        for (int x = 0; x < W; ++x) { p.at(x,0) = p.at(x,1); p.at(x,H-1) = p.at(x,H-2); }
    }

    void calcDivergence() {
        divergence.clear();
        for (int y = 1; y < H-1; ++y) for (int x = 1; x < W-1; ++x)
            divergence.at(x,y) = .5f * (vx.at(x+1,y) - vx.at(x-1,y) + vy.at(x,y+1) - vy.at(x,y-1));
        if (obstacleEnabled) for (int k = 0; k < N; ++k)
            if (obstacle[k] > .30f) divergence.v[k] = 0.f;
    }

    void project() {
        velocityBoundaries();
        calcDivergence();
        pressure.clear(); tmpPressure.clear();
        Field *p = &pressure, *out = &tmpPressure;
        for (int it = 0; it < pressureIterations; ++it) {
            for (int y = 1; y < H-1; ++y) for (int x = 1; x < W-1; ++x) {
                int k = ix(x,y);
                if (obstacleEnabled && obstacle[k] > .30f) { out->v[k] = 0.f; continue; }
                auto neighbor = [&](int nx, int ny) {
                    return (obstacleEnabled && obstacle[ix(nx,ny)] > .30f) ? p->at(x,y) : p->at(nx,ny);
                };
                out->at(x,y) = .25f * (neighbor(x-1,y) + neighbor(x+1,y) +
                                      neighbor(x,y-1) + neighbor(x,y+1) - divergence.at(x,y));
            }
            pressureBoundaries(*out);
            std::swap(p, out);
        }
        if (p != &pressure) pressure.v = p->v;
        for (int y = 1; y < H-1; ++y) for (int x = 1; x < W-1; ++x) {
            vx.at(x,y) -= .5f * (pressure.at(x+1,y) - pressure.at(x-1,y));
            vy.at(x,y) -= .5f * (pressure.at(x,y+1) - pressure.at(x,y-1));
        }
        velocityBoundaries();
        calcDivergence();
        double sum = 0;
        for (int y = 1; y < H-1; ++y) for (int x = 1; x < W-1; ++x)
            sum += double(divergence.at(x,y)) * divergence.at(x,y);
        divergenceRms = std::sqrt(sum / ((W-2)*(H-2)));
    }

    void vorticity(float dt) {
        curl.clear(); gradX.clear(); gradY.clear();
        for (int y = 1; y < H-1; ++y) for (int x = 1; x < W-1; ++x)
            curl.at(x,y) = .5f * (vy.at(x+1,y) - vy.at(x-1,y) - vx.at(x,y+1) + vx.at(x,y-1));
        double sum = 0;
        for (int y = 1; y < H-1; ++y) for (int x = 1; x < W-1; ++x) {
            float c = curl.at(x,y); sum += double(c)*c;
            gradX.at(x,y) = .5f * (std::abs(curl.at(x+1,y)) - std::abs(curl.at(x-1,y)));
            gradY.at(x,y) = .5f * (std::abs(curl.at(x,y+1)) - std::abs(curl.at(x,y-1)));
        }
        curlRms = std::sqrt(sum / ((W-2)*(H-2)));
        for (int k = 0; k < N; ++k) {
            float inv = 1.f / (std::hypot(gradX.v[k], gradY.v[k]) + 1.e-5f);
            float nx = gradX.v[k] * inv, ny = gradY.v[k] * inv;
            float activity = clamp(.12f + density.v[k] * .55f + fuel.v[k] * .33f, .12f, 1.f);
            vx.v[k] += clamp(ny * curl.v[k] * 22.f, -80.f, 80.f) * activity * dt;
            vy.v[k] += clamp(-nx * curl.v[k] * 22.f, -80.f, 80.f) * activity * dt;
        }
    }

    void step(float dt) {
        elapsed += dt;
        addSource(dt);
        combust(dt);
        for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
            int k = ix(x,y);
            vy.v[k] -= (temp.v[k] * 34.f - density.v[k] * 4.5f) * dt;
            vx.v[k] += std::sin(y * .073f + elapsed * .61f) * (density.v[k] + fuel.v[k] * .45f) * dt * .30f;
        }
        oldVx.v = vx.v; oldVy.v = vy.v;
        advect(vx, oldVx, oldVx, oldVy, dt);
        advect(vy, oldVy, oldVx, oldVy, dt);
        vorticity(dt);
        project();
        for (Field *f : {&density, &temp, &fuel, &burn}) {
            tempField.v = f->v;
            advect(*f, tempField, vx, vy, dt);
        }
        if (obstacleEnabled) {
            for (int k = 0; k < N; ++k) if (obstacle[k] > .30f) {
                density.v[k] = temp.v[k] = fuel.v[k] = burn.v[k] = 0.f;
                vx.v[k] = vy.v[k] = 0.f;
            }
        }
        for (int k = 0; k < N; ++k) {
            vx.v[k] *= .992f;
            vy.v[k] *= .992f;
            // Lifetime 1.2x: pow(0.996, 1/1.2) ≈ 0.996664.
            density.v[k] = clamp(density.v[k] * .996664f, 0.f, 4.f);
            temp.v[k] = clamp(temp.v[k] * .990f, 0.f, maxTemp);
            fuel.v[k] = clamp(fuel.v[k] * .996f, 0.f, maxFuel);
            burn.v[k] *= .90f;
        }
        for (int x = 0; x < W; ++x) {
            density.at(x,H-1) *= .80f;
            temp.at(x,H-1) *= .84f;
            fuel.at(x,H-1) *= .82f;
        }
    }
};

std::string readFile(const std::filesystem::path &path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Shader not found: " + path.string());
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

GLuint compile(GLenum type, const std::string &source) {
    GLuint shader = glCreateShader(type);
    const char *ptr = source.c_str();
    glShaderSource(shader, 1, &ptr, nullptr);
    glCompileShader(shader);
    GLint ok; glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        GLint length = 0; glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
        std::string log(std::max(1, length), '\0');
        glGetShaderInfoLog(shader, length, nullptr, log.data());
        glDeleteShader(shader);
        throw std::runtime_error("GLSL compilation failed:\n" + log);
    }
    return shader;
}

struct Renderer {
    GLuint program = 0, vao = 0, vbo = 0;
    std::array<GLuint,4> textures{};
    Renderer(const std::filesystem::path &dir) {
        GLuint vs = compile(GL_VERTEX_SHADER, readFile(dir / "quad.vert"));
        GLuint fs = compile(GL_FRAGMENT_SHADER, readFile(dir / "fire_smoke.frag"));
        program = glCreateProgram();
        glAttachShader(program, vs); glAttachShader(program, fs);
        glLinkProgram(program);
        glDeleteShader(vs); glDeleteShader(fs);
        GLint ok; glGetProgramiv(program, GL_LINK_STATUS, &ok);
        if (!ok) {
            GLint len; glGetProgramiv(program, GL_INFO_LOG_LENGTH, &len);
            std::string log(std::max(1,len), '\0');
            glGetProgramInfoLog(program, len, nullptr, log.data());
            throw std::runtime_error("GLSL linking failed: " + log);
        }
        const float vertices[] = {
            -1,-1, 0,1,  1,-1, 1,1, -1,1, 0,0,
            -1, 1, 0,0,  1,-1, 1,1,  1,1, 1,0
        };
        glGenVertexArrays(1, &vao); glBindVertexArray(vao);
        glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4*sizeof(float), nullptr);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4*sizeof(float), reinterpret_cast<void*>(2*sizeof(float)));
        glEnableVertexAttribArray(0); glEnableVertexAttribArray(1);
        glGenTextures(4, textures.data());
        glUseProgram(program);
        const char *names[] = {"density_tex", "temperature_tex", "fuel_tex", "burn_tex"};
        for (int i = 0; i < 4; ++i) {
            glActiveTexture(GL_TEXTURE0 + i);
            glBindTexture(GL_TEXTURE_2D, textures[i]);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, W, H, 0, GL_RED, GL_FLOAT, nullptr);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glUniform1i(glGetUniformLocation(program, names[i]), i);
        }
        glUniform2f(glGetUniformLocation(program, "texel_size"), 1.f/W, 1.f/H);
    }
    ~Renderer() {
        glDeleteTextures(4, textures.data());
        glDeleteBuffers(1, &vbo); glDeleteVertexArrays(1, &vao); glDeleteProgram(program);
    }
    void draw(const Simulation &s, int width, int height, bool vol, float volStrength) {
        const Field *fields[] = {&s.density, &s.temp, &s.fuel, &s.burn};
        for (int i=0; i<4; ++i) {
            glActiveTexture(GL_TEXTURE0+i);
            glBindTexture(GL_TEXTURE_2D, textures[i]);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, W, H, GL_RED, GL_FLOAT, fields[i]->v.data());
        }
        glUseProgram(program);
        glUniform1i(glGetUniformLocation(program, "debug_mode"), s.debugMode);
        glUniform1i(glGetUniformLocation(program, "show_fire"), s.showFire ? 1 : 0);
        glUniform1i(glGetUniformLocation(program, "volumetric_enabled"), vol ? 1 : 0);
        glUniform1f(glGetUniformLocation(program, "volumetric_strength"), volStrength);
        glViewport(0,0,std::max(width,1),std::max(height,1));
        glClearColor(4.f/255,5.f/255,7.f/255,1.f);
        glClear(GL_COLOR_BUFFER_BIT);
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, 6);
    }
};

// Text glyphs are rasterized once into a transparent overlay and the same
// alpha mask becomes the fluid's solid boundary. Shadow is visual-only.
struct TitleOverlay {
    GLuint program = 0, texture = 0, vao = 0, vbo = 0;
    static constexpr int CanvasW = 1280, CanvasH = 720;

    static void blitText(SDL_Surface *canvas, TTF_Font *font, const char *text,
                         SDL_Color color, int centerY, int xOffset = 0, int yOffset = 0) {
        SDL_Surface *glyph = TTF_RenderUTF8_Blended(font, text, color);
        if (!glyph) throw std::runtime_error(TTF_GetError());
        SDL_Rect at{(CanvasW - glyph->w) / 2 + xOffset,
                    centerY - glyph->h / 2 + yOffset, glyph->w, glyph->h};
        SDL_BlitSurface(glyph, nullptr, canvas, &at);
        SDL_FreeSurface(glyph);
    }

    explicit TitleOverlay(const std::filesystem::path &assetDir, Simulation &sim) {
        if (TTF_Init() < 0) throw std::runtime_error(TTF_GetError());
        const auto fontPath = (assetDir / "Montserrat.ttf").string();
        TTF_Font *big = TTF_OpenFont(fontPath.c_str(), 74);
        const auto subtitlePath = (assetDir / "CormorantGaramond.ttf").string();
        TTF_Font *small = TTF_OpenFont(subtitlePath.c_str(), 43);
        if (!big || !small) throw std::runtime_error("Unable to open Montserrat: " + fontPath);
        TTF_SetFontStyle(big, TTF_STYLE_BOLD);
        SDL_Surface *canvas = SDL_CreateRGBSurfaceWithFormat(0, CanvasW, CanvasH, 32, SDL_PIXELFORMAT_RGBA32);
        SDL_Surface *mask = SDL_CreateRGBSurfaceWithFormat(0, CanvasW, CanvasH, 32, SDL_PIXELFORMAT_RGBA32);
        if (!canvas || !mask) throw std::runtime_error(SDL_GetError());
        SDL_FillRect(canvas, nullptr, SDL_MapRGBA(canvas->format,0,0,0,0));
        SDL_FillRect(mask, nullptr, SDL_MapRGBA(mask->format,0,0,0,0));

        const SDL_Color shadow{12,4,3,180}, warm{255,223,184,244}, sub{235,194,158,237};
        blitText(canvas,big,"PROMETHEUS",shadow,327,3,4);
        blitText(canvas,small,"The Usurper of Fire",shadow,410,2,3);
        blitText(canvas,big,"PROMETHEUS",warm,327);
        blitText(canvas,small,"The Usurper of Fire",sub,410);
        blitText(mask,big,"PROMETHEUS",SDL_Color{255,255,255,255},327);
        blitText(mask,small,"The Usurper of Fire",SDL_Color{255,255,255,255},410);

        SDL_LockSurface(mask);
        for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
            // Use the maximum alpha over the corresponding few full-res pixels.
            int cx = (x * CanvasW + CanvasW / (2*W)) / W;
            int cy = (y * CanvasH + CanvasH / (2*H)) / H;
            Uint8 peak = 0;
            for (int oy=-2; oy<=2; ++oy) for (int ox=-2; ox<=2; ++ox) {
                int px = std::clamp(cx+ox,0,CanvasW-1), py = std::clamp(cy+oy,0,CanvasH-1);
                const auto row = reinterpret_cast<const Uint32*>(static_cast<const Uint8*>(mask->pixels) + py * mask->pitch);
                Uint8 r,g,b,a;
                SDL_GetRGBA(row[px],mask->format,&r,&g,&b,&a);
                peak = std::max(peak,a);
            }
            sim.obstacle[ix(x,y)] = peak / 255.f;
        }
        SDL_UnlockSurface(mask);

        auto vertex = compile(GL_VERTEX_SHADER,readFile(assetDir.parent_path() / "shaders/quad.vert"));
        auto fragment = compile(GL_FRAGMENT_SHADER,readFile(assetDir.parent_path() / "shaders/title.frag"));
        program = glCreateProgram();
        glAttachShader(program,vertex); glAttachShader(program,fragment); glLinkProgram(program);
        glDeleteShader(vertex); glDeleteShader(fragment);
        GLint linked = 0; glGetProgramiv(program,GL_LINK_STATUS,&linked);
        if (!linked) throw std::runtime_error("Title shader link failed");
        const float vertices[] = {-1,-1,0,1, 1,-1,1,1, -1,1,0,0,
                                  -1, 1,0,0, 1,-1,1,1,  1,1,1,0};
        glGenVertexArrays(1,&vao); glBindVertexArray(vao);
        glGenBuffers(1,&vbo); glBindBuffer(GL_ARRAY_BUFFER,vbo);
        glBufferData(GL_ARRAY_BUFFER,sizeof(vertices),vertices,GL_STATIC_DRAW);
        glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,4*sizeof(float),nullptr);
        glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE,4*sizeof(float),reinterpret_cast<void*>(2*sizeof(float)));
        glEnableVertexAttribArray(0); glEnableVertexAttribArray(1);
        glGenTextures(1,&texture); glActiveTexture(GL_TEXTURE4); glBindTexture(GL_TEXTURE_2D,texture);
        glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA8,CanvasW,CanvasH,0,GL_RGBA,GL_UNSIGNED_BYTE,canvas->pixels);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
        glUseProgram(program); glUniform1i(glGetUniformLocation(program,"title_tex"),4);

        SDL_FreeSurface(canvas); SDL_FreeSurface(mask);
        TTF_CloseFont(big); TTF_CloseFont(small);
    }
    ~TitleOverlay() {
        glDeleteTextures(1,&texture); glDeleteBuffers(1,&vbo);
        glDeleteVertexArrays(1,&vao); glDeleteProgram(program);
        TTF_Quit();
    }
    void draw() const {
        glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
        glUseProgram(program); glActiveTexture(GL_TEXTURE4); glBindTexture(GL_TEXTURE_2D,texture);
        glBindVertexArray(vao); glDrawArrays(GL_TRIANGLES,0,6);
        glDisable(GL_BLEND);
    }
};

std::filesystem::path shaderDir(const char *argv0) {
    const auto executableDir = std::filesystem::absolute(argv0).parent_path();
    if (std::filesystem::exists(executableDir / "shaders/fire_smoke.frag")) return executableDir / "shaders";
    if (std::filesystem::exists("shaders/fire_smoke.frag")) return "shaders";
    return executableDir / "shaders";
}
}

int main(int argc, char **argv) {
    (void)argc;
    SDL_Window *window = nullptr;
    SDL_GLContext context = nullptr;
    try {
        if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0) throw std::runtime_error(SDL_GetError());
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
        SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
        window = SDL_CreateWindow("Smoke & Flame C++ v1.2", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                  1280, 720, SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);
        if (!window) throw std::runtime_error(SDL_GetError());
        context = SDL_GL_CreateContext(window);
        if (!context) throw std::runtime_error(SDL_GetError());
        glewExperimental = GL_TRUE;
        if (glewInit() != GLEW_OK) throw std::runtime_error("Failed to initialize GLEW");
        glGetError(); // GLEW may emit GL_INVALID_ENUM with core profiles.
        SDL_GL_SetSwapInterval(1);
        {
            Renderer renderer(shaderDir(argv[0]));
            Simulation sim;
            TitleOverlay title(shaderDir(argv[0]).parent_path() / "assets", sim);
            bool running = true, paused = false, vol = true;
            float volStrength = .38f;
            auto previous = SDL_GetPerformanceCounter();
            double titleTimer = 0, frames = 0;
            while (running) {
                const auto now = SDL_GetPerformanceCounter();
                float dt = std::min(float(double(now-previous) / SDL_GetPerformanceFrequency()), 1.f/30.f);
                previous = now;
                SDL_Event event;
                while (SDL_PollEvent(&event)) {
                    if (event.type == SDL_QUIT) running = false;
                    if (event.type == SDL_KEYDOWN && !event.key.repeat) {
                        switch (event.key.keysym.sym) {
                            case SDLK_ESCAPE: running = false; break;
                            case SDLK_SPACE: paused = !paused; break;
                            case SDLK_r: sim.reset(); break;
                            case SDLK_f: sim.showFire = !sim.showFire; break;
                            case SDLK_t: sim.debugMode = sim.debugMode == 1 ? 0 : 1; break;
                            case SDLK_g: sim.debugMode = sim.debugMode == 2 ? 0 : 2; break;
                            case SDLK_l: vol = !vol; break;
                            case SDLK_o: sim.obstacleEnabled = !sim.obstacleEnabled; break;
                            case SDLK_LEFTBRACKET: volStrength = std::max(0.f,volStrength-.05f); break;
                            case SDLK_RIGHTBRACKET: volStrength = std::min(1.f,volStrength+.05f); break;
                            case SDLK_EQUALS: case SDLK_PLUS: case SDLK_KP_PLUS: sim.intensity = std::min(3.f,sim.intensity+.1f); break;
                            case SDLK_MINUS: case SDLK_KP_MINUS: sim.intensity = std::max(.2f,sim.intensity-.1f); break;
                            default: break;
                        }
                    }
                }
                int winW, winH, drawW, drawH;
                SDL_GetWindowSize(window,&winW,&winH);
                SDL_GL_GetDrawableSize(window,&drawW,&drawH);
                if (!paused) {
                    int mx,my;
                    if (SDL_GetMouseState(&mx,&my) & SDL_BUTTON(SDL_BUTTON_LEFT))
                        sim.addMouse(float(mx),float(my),winW,winH,dt);
                    sim.step(dt);
                }
                renderer.draw(sim,drawW,drawH,vol,volStrength);
                title.draw();
                SDL_GL_SwapWindow(window);
                titleTimer += dt; ++frames;
                if (titleTimer >= .5) {
                    std::ostringstream title;
                    title.setf(std::ios::fixed); title.precision(1);
                    title << "Smoke & Flame C++ v1.2 | " << frames/titleTimer << " FPS";
                    title.precision(4);
                    title << " | div " << sim.divergenceRms << " curl " << sim.curlRms;
                    title.precision(2);
                    title << " | intensity " << sim.intensity << " | " << (sim.showFire ? "FIRE" : "SMOKE")
                          << " | " << (vol ? "VOL" : "FLAT") << (paused ? " | PAUSED" : "");
                    SDL_SetWindowTitle(window,title.str().c_str());
                    titleTimer = frames = 0;
                }
            }
        }
        SDL_GL_DeleteContext(context); SDL_DestroyWindow(window); SDL_Quit();
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "Smoke & Flame: " << e.what() << '\n';
        if (context) SDL_GL_DeleteContext(context);
        if (window) SDL_DestroyWindow(window);
        SDL_Quit(); return 1;
    }
}
