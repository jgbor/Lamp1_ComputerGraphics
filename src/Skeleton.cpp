#include "framework.h"

// vertex shader in GLSL
const char *vertexSource = R"(
	#version 330
    precision highp float;

	uniform vec3 wLookAt, wRight, wUp;          // pos of eye

	layout(location = 0) in vec2 cCamWindowVertex;	// Attrib Array 0
	out vec3 p;

	void main() {
		gl_Position = vec4(cCamWindowVertex, 0, 1);
		p = wLookAt + wRight * cCamWindowVertex.x + wUp * cCamWindowVertex.y;
	}
)";

// fragment shader in GLSL
const char *fragmentSource = R"(
	#version 330
    precision highp float;

	struct Material {
		vec3 ka, kd, ks;
		float shininess;
		vec3 F0;
		int rough, reflective;
	};

	struct Light {
		vec3 direction,position;
		vec3 Le, La;
	};

	struct Gomb {
		vec3 center;
		float radius;
	};

    struct Henger {
        vec3 top, bottom, middle, normal;
        float radius, height;
    };

    struct Sik {
        vec3 point, normal;
    };

    struct Parabola {
        vec3 focus;
        float height;
    };

	struct Hit {
		float t;
		vec3 position, normal;
		int mat;	// material index
	};

	struct Ray {
		vec3 start, dir;
	};


	uniform vec3 wEye;
    uniform int nLights;
	uniform Light lights[2];
	uniform Material materials[2];  // diffuse, specular, ambient ref
	uniform int nGombok;
	uniform Gomb gombok[3];
    uniform Sik sik;
    uniform Parabola parabola;
    uniform int nHengerek;
    uniform Henger hengerek[3];

	in  vec3 p;					// point on camera window corresponding to the pixel
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

    const float epsilon = 0.0001f;
	const int maxdepth = 5;

	Hit intersectGomb(const Gomb object, const Ray ray) {
		Hit hit;
		hit.t = -1;
		vec3 dist = ray.start - object.center;
		float a = dot(ray.dir, ray.dir);
		float b = dot(dist, ray.dir) * 2.0f;
		float c = dot(dist, dist) - object.radius * object.radius;
		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrt(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;	// t1 >= t2 for sure
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return hit;
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = (hit.position - object.center) / object.radius;
		return hit;
	}

    float fabs(const float f){
        if (f<0) return f * -1;
        return f;
    }

    Hit intersectSik(const Sik s,const Ray ray) {
        Hit hit;
        float NdotV = dot(s.normal, ray.dir);
        if (fabs(NdotV) < epsilon) return hit;
        float t = dot(s.normal, s.point - ray.start) / NdotV;
        if (t < epsilon) return hit;
        hit.t = t;
        hit.position = ray.start + ray.dir * hit.t;
        hit.normal = s.normal;
        if (dot(hit.normal, ray.dir) > 0) hit.normal = hit.normal * (-1);
        return hit;
    }

    Hit intersectHenger(const Henger henger, Ray ray){
        Hit hit;
        hit.t = -1;
		vec2 dist = ray.start.xz - henger.bottom.xz;
		float a = dot(ray.dir.xz, ray.dir.xz);
		float b = dot(dist, ray.dir.xz) * 2.0f;
		float c = dot(dist, dist) - henger.radius * henger.radius;
		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrt(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;
		float t2 = (-b - sqrt_discr) / 2.0f / a;
        if ((ray.start + ray.dir * t1).y<henger.bottom.y || (ray.start + ray.dir * t1).y> henger.top.y){
            t1=-1;
        }
        if ((ray.start + ray.dir * t2).y<henger.bottom.y || (ray.start + ray.dir * t2).y> henger.top.y){
            t2=-1;
        }
		if (t1 <= 0 && t2 <= 0){
            return hit;
        }else{
		    hit.t = (t2 > 0) ? t2 : t1;
        }
        hit.position = ray.start + ray.dir * hit.t;
		hit.normal = hit.position - henger.bottom;
        hit.normal.y = 0;
        normalize(hit.normal);
        return hit;
    }

    Hit intersectParabola(const Parabola para, Ray ray){
        Hit hit;
		hit.t = -1;
        ray.start -= para.focus;
        //Az a,b,c valtozok kiszamitasahoz  Csala Balint konzultaciojanak pptjeben levo kepleteket hasznaltam fel
		float a = dot(ray.dir.xz,ray.dir.xz);
        float b = 2.0f * dot(ray.start.xz,ray.dir.xz) - ray.dir.y;
        float c = dot(ray.start.xz,ray.start.xz) - ray.start.y;
		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrt(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;
		float t2 = (-b - sqrt_discr) / 2.0f / a;
        if ((ray.start + ray.dir * t1).y > para.height){
            t1=-1;
        }
        if ((ray.start + ray.dir * t2).y > para.height){
            t2=-1;
        }
		if (t1 <= 0 && t2 <= 0){
            return hit;
        }else{
		    hit.t = (t2 > 0) ? t2 : t1;
        }
		hit.position = ray.start + ray.dir * hit.t + para.focus;
		hit.normal = vec3(1,hit.position.x*2.0f,0) * vec3(0,hit.position.z*2.0f,1);
        normalize(hit.normal);
		return hit;
    }

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		bestHit.t = -1;
        Hit hit;
        hit.mat=0;
		for (int o = 0; o < nGombok; o++) {
			hit = intersectGomb(gombok[o], ray);
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
        for (int i = 0; i < nHengerek; i++) {
			hit = intersectHenger(hengerek[i], ray);
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
            Sik vagosik;
            vagosik.normal=hengerek[i].middle;
            vagosik.point=hengerek[i].top;
            Hit sikkal = intersectSik(vagosik,ray);
            if (length(hengerek[i].top - sikkal.position)<hengerek[i].radius){
                hit=sikkal;
            }
            if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
            vagosik.point=hengerek[i].bottom;
            sikkal = intersectSik(vagosik,ray);
            if (length(hengerek[i].bottom - sikkal.position)<hengerek[i].radius){
                sikkal.position.y=sikkal.position.y+hengerek[i].height;
                hit=sikkal;
            }
            if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
        hit = intersectParabola(parabola, ray);
        if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t)){
            bestHit=hit;
        }
        hit = intersectSik(sik, ray);
        if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t)){
            hit.mat=1;
            bestHit=hit;
        }
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	bool shadowIntersect(Ray ray) {	// for directional lights
		for (int o = 0; o < nGombok; o++) if (intersectGomb(gombok[o], ray).t > 0) return true;
		for (int o = 0; o < nHengerek; o++) if (intersectHenger(hengerek[o], ray).t > 0) return true;
        if (intersectParabola(parabola, ray).t>0) return true;
		return false;
	}

	vec3 Fresnel(vec3 F0, float cosTheta) {
		return F0 + (vec3(1, 1, 1) - F0) * pow(cosTheta, 5);
	}

	vec3 trace(Ray ray) {
		vec3 weight = vec3(1, 2, 1);
		vec3 outRadiance = vec3(0, 0, 0);
		for(int d = 0; d < maxdepth; d++) {
			Hit hit = firstIntersect(ray);
			if (hit.t < 0) return weight * lights[0].La;
			if (materials[hit.mat].rough == 1) {
				outRadiance += weight * materials[hit.mat].ka * lights[0].La;
				Ray shadowRay;
				shadowRay.start = hit.position + hit.normal * epsilon;
				shadowRay.dir = lights[0].direction;
				float cosTheta = dot(hit.normal, lights[0].direction);
				if (cosTheta > 0 && !shadowIntersect(shadowRay)) {
					outRadiance += weight * lights[0].Le * materials[hit.mat].kd * cosTheta;
					vec3 halfway = normalize(-ray.dir + lights[0].direction);
					float cosDelta = dot(hit.normal, halfway);
					if (cosDelta > 0) outRadiance += weight * lights[0].Le * materials[hit.mat].ks * pow(cosDelta, materials[hit.mat].shininess);
				}
			}

			if (materials[hit.mat].reflective == 1) {
				weight *= Fresnel(materials[hit.mat].F0, dot(-ray.dir, hit.normal));
				ray.start = hit.position + hit.normal * epsilon;
				ray.dir = reflect(ray.dir, hit.normal);
			} else return outRadiance;
		}
	}

	void main() {
		Ray ray;
        for (int i=0; i<nLights; i++){
		    ray.start = wEye + lights[i].position;
		    ray.dir = normalize(p - (wEye + lights[i].position));
		    fragmentColor = vec4(trace(ray), 1);
        }
	}
)";

vec4 qmul(vec4 q1, vec4 q2) {
    vec3 d1(q1.x, q1.y, q1.z), d2(q2.x, q2.y, q2.z);
    return vec4((d2 * q1.w + d1 * q2.w + cross(d1, d2)).x, (d2 * q1.w + d1 * q2.w + cross(d1, d2)).y, (d2 * q1.w + d1 * q2.w + cross(d1, d2)).z,
                q1.w * q2.w - dot(d1, d2));
}
vec4 quaternion(float ang, vec3 axis) {
    vec3 d = normalize(axis) * sinf(ang / 2);
    return vec4(d.x, d.y, d.z, cosf(ang / 2));
}
vec3 Rotate(vec3 u, vec4 q) {
    vec4 qinv(-q.x, -q.y, -q.z, q.w);
    vec4 qr = qmul(qmul(q, vec4(u.x, u.y, u.z, 0)), qinv);
    return vec3(qr.x, qr.y, qr.z);
}

struct Material {
    vec3 ka, kd, ks;
    float  shininess;
    vec3 F0;
    int rough, reflective;
};

struct RoughMaterial : public Material {
    RoughMaterial(vec3 _kd, vec3 _ks, float _shininess) {
        ka = _kd * M_PI;
        kd = _kd;
        ks = _ks;
        shininess = _shininess;
        rough = true;
        reflective = false;
    }
};

struct Hit {
    float t;
    vec3 position, normal;
    Material * material;
    Hit() { t = -1; }
};

struct Gomb {
    vec3 center;
    float radius;

    Gomb(const vec3& _center, float _radius){
        center = _center;
        radius = _radius;
    }
};

struct Henger{
    vec3 top, bottom;
    float radius, height;
    vec3 normal, middle;

    Henger(const vec3& bottom, const float height, const  float radius){
        this->bottom = bottom;
        this->height=height;
        this->top = vec3(bottom.x,bottom.y+height,bottom.z);
        this->radius=radius;
        middle=top-bottom;
    }
};

struct Sik {
    vec3 point,normal;

    Sik(){}

    Sik(const vec3& _point, const vec3& _normal) {
        normal = normalize(_normal);
        point=_point;
    }
};

struct Parabola{
    vec3 focus;
    float height;

    Parabola(){}

    Parabola(const vec3& fcs,  const float h){
        focus=fcs;
        height=h;
    }
};

struct Camera {
    vec3 eye, lookat, right, up;
    float fov;
public:
    void set(vec3 _eye, vec3 _lookat, vec3 vup, float _fov) {
        eye = _eye;
        lookat = _lookat;
        fov = _fov;
        vec3 w = eye - lookat;
        float f = length(w);
        right = normalize(cross(vup, w)) * f * tanf(fov / 2);
        up = normalize(cross(w, right)) * f * tanf(fov / 2);
    }
    void Animate(float dt) {
        eye = vec3((eye.x - lookat.x) * cos(dt) + (eye.z - lookat.z) * sin(dt) + lookat.x,
                   eye.y,
                   -(eye.x - lookat.x) * sin(dt) + (eye.z - lookat.z) * cos(dt) + lookat.z);
        set(eye, lookat, up, fov);
    }
};

struct Light {
    vec3 position,direction;
    vec3 Le, La;
    Light(const vec3& pos, const vec3& _direction,const vec3& _Le,const vec3& _La) {
        direction = normalize(_direction);
        position = pos;
        Le = _Le; La = _La;
    }
};

class Shader : public GPUProgram {
public:
    void setUniformMaterials(const std::vector<Material*>& materials) {
        char name[256];
        for (unsigned int mat = 0; mat < materials.size(); mat++) {
            sprintf(name, "materials[%d].ka", mat); setUniform(materials[mat]->ka, name);
            sprintf(name, "materials[%d].kd", mat); setUniform(materials[mat]->kd, name);
            sprintf(name, "materials[%d].ks", mat); setUniform(materials[mat]->ks, name);
            sprintf(name, "materials[%d].shininess", mat); setUniform(materials[mat]->shininess, name);
            sprintf(name, "materials[%d].F0", mat); setUniform(materials[mat]->F0, name);
            sprintf(name, "materials[%d].rough", mat); setUniform(materials[mat]->rough, name);
            sprintf(name, "materials[%d].reflective", mat); setUniform(materials[mat]->reflective, name);
        }
    }

    void setUniformLight(const std::vector<Light*>& lights) {
        setUniform((int)lights.size(),"nLights");
        char name[256];
        for(unsigned int i=0; i<lights.size();i++) {
            sprintf(name,"lights[%d].La",i); setUniform(lights[i]->La, name);
            sprintf(name,"lights[%d].Le",i); setUniform(lights[i]->Le, name);
            sprintf(name,"lights[%d].direction",i); setUniform(lights[i]->direction, name);
            sprintf(name,"lights[%d].position",i); setUniform(lights[i]->position,name);
        }
    }

    void setUniformCamera(const Camera& camera) {
        setUniform(camera.eye, "wEye");
        setUniform(camera.lookat, "wLookAt");
        setUniform(camera.right, "wRight");
        setUniform(camera.up, "wUp");
    }

    void setUniformGomb(const std::vector<Gomb*>& gombok) {
        setUniform((int)gombok.size(), "nGombok");
        char name[256];
        for (unsigned int o = 0; o < gombok.size(); o++) {
            sprintf(name, "gombok[%d].center", o);  setUniform(gombok[o]->center, name);
            sprintf(name, "gombok[%d].radius", o);  setUniform(gombok[o]->radius, name);
        }
    }

    void setUniformSik(const Sik& sik){
        setUniform(sik.normal,"sik.normal");
        setUniform(sik.point,"sik.point");
    }

    void setUniformHenger(const std::vector<Henger*>& hengerek){
        setUniform((int)hengerek.size(),"nHengerek");
        char name[256];
        for (int i = 0; i<hengerek.size();i++){
            sprintf(name, "hengerek[%d].top",i); setUniform(hengerek[i]->top, name);
            sprintf(name, "hengerek[%d].bottom",i); setUniform(hengerek[i]->bottom, name);
            sprintf(name, "hengerek[%d].radius",i); setUniform(hengerek[i]->radius, name);
            sprintf(name, "hengerek[%d].normal",i); setUniform(hengerek[i]->normal, name);
            sprintf(name, "hengerek[%d].middle",i); setUniform(hengerek[i]->middle, name);
            sprintf(name, "hengerek[%d].height",i); setUniform(hengerek[i]->height, name);
        }
    }

    void setUniformParabola(const Parabola& parabola){
        setUniform(parabola.height,"parabola.height");
        setUniform(parabola.focus,"parabola.focus");
    }
};

class Scene {
    std::vector<Gomb*> gombok;
    Sik sik;
    Parabola parabola;
    std::vector<Henger*> hengerek;
    std::vector<Light *> lights;
    Camera camera;
    std::vector<Material *> materials;
public:
    void build() {
        vec3 eye = vec3(0, 0, 4);
        vec3 vup = vec3(0, 1, 0);
        vec3 lookat = vec3(0, 0, 0);
        float fov = 45 * (float)M_PI / 180;
        camera.set(eye, lookat, vup, fov);

        lights.push_back(new Light(0, vec3(1, 1, 1), vec3(1, 1, 1), vec3(0.2f, 0.05f, 0.05f)));

        vec3 kd1(0.1f, 0.1f, 0.8f), ks1(5, 5, 5);
        vec3 kd2(0.5f,0.457f,0.5f), ks2(1,1,1);
        materials.push_back(new RoughMaterial(kd1, ks1, 75));
        materials.push_back(new RoughMaterial(kd2, ks2, 500));

        float x=0,y=-0.5f,z=0;
        float r=0.1f;
        sik = Sik(vec3(x,y,z),vec3(0,1,0));
        for (int i = 0; i < 3; i++) {
            if(i==0) {
                hengerek.push_back(new Henger(vec3(x, y, z), 0.1f, r*5.0f));
                y=y+hengerek[i]->height;
            }else{
                hengerek.push_back(new Henger(vec3(x, y, z), 0.6f, r/2.0f));
                y=y+hengerek[i]->height+r*0.9f;
            }
            gombok.push_back(new Gomb(vec3(x, y, z), r));
        }
        y=y+r;
        parabola = Parabola(vec3(x,y,z),0.25f);
        lights.push_back(new Light(vec3(x,y,z), vec3(1, 1, 1), vec3(5, 5, 5), vec3(0.2f, 0.05f, 0.05f)));
    }

    void setUniform(Shader& shader) {
        shader.setUniformGomb(gombok);
        shader.setUniformMaterials(materials);
        shader.setUniformLight(lights);
        shader.setUniformCamera(camera);
        shader.setUniformSik(sik);
        shader.setUniformHenger(hengerek);
        shader.setUniformParabola(parabola);
    }

    void Animate(float dt) { camera.Animate(dt); }
};

Shader shader; // vertex and fragment shaders
Scene scene;

class FullScreenTexturedQuad {
    unsigned int vao = 0;
public:
    void create() {
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        unsigned int vbo;
        glGenBuffers(1, &vbo);

        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
    }

    void Draw() {
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    }
};

FullScreenTexturedQuad fullScreenTexturedQuad;
// Initialization, create an OpenGL context
void onInitialization() {
    glViewport(0, 0, windowWidth, windowHeight);
    scene.build();
    fullScreenTexturedQuad.create();

    // create program for the GPU
    shader.create(vertexSource, fragmentSource, "fragmentColor");
    shader.Use();
}

// Window has become invalid: Redraw
void onDisplay() {
    static int nFrames = 0;
    nFrames++;
    static long tStart = glutGet(GLUT_ELAPSED_TIME);
    long tEnd = glutGet(GLUT_ELAPSED_TIME);

    glClearColor(1.0f, 0.5f, 0.8f, 1.0f);							// background color
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen

    scene.setUniform(shader);
    fullScreenTexturedQuad.Draw();

    glutSwapBuffers();
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
    scene.Animate(0.01f);
    glutPostRedisplay();
}
