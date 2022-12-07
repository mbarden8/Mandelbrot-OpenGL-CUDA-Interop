// opengl stuff
#include <glad/glad.h>
#include <GLFW/glfw3.h>

// cuda stuff
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// my stuff
#include "mandelbrot.cuh"
#include "shader.h"

// visual studio is annoying stuff
#include "device_launch_parameters.h"

// standard stuff
#include <iostream>

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);

// some settings
const int SCR_WIDTH = 1920;
const int SCR_HEIGHT = 1080;
// for mandelbrot iterations
const int MAX_ITERATIONS = 200;

int main(int argc, char** argv)
{
    /* Initialize glfw window */
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Mandelbrot", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    /* Create our shader using shader.h file */
    Shader ourShader("../../Mandelbrot-OpenGLCudaInterop/texture.vs", "../../Mandelbrot-OpenGLCudaInterop/texture.fs");

    /* Create our vertices for our screen and texture */
    float vertices[] = {
        // positions          // texture coords
         1.0f,  1.0f, 0.0f,   1.0f, 1.0f,       // top right
         1.0f, -1.0f, 0.0f,   1.0f, 0.0f,       // bottom right
        -1.0f, -1.0f, 0.0f,   0.0f, 0.0f,       // bottom left
        -1.0f,  1.0f, 0.0f,   0.0f, 1.0f        // top left 
    };
    unsigned int indices[] = {
        0, 1, 3, // first triangle
        1, 2, 3  // second triangle
    };

    /* Vertex buffer and vertex array stuff */
    unsigned int VBO, VAO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    /* Create our blank texture to draw the mandelbrot results to */
    unsigned int texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    glGenerateMipmap(GL_TEXTURE_2D);

    /* Register our texture with cuda for interop */
    cudaGraphicsResource_t textureResource;
    cudaGraphicsGLRegisterImage(&textureResource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
    struct cudaResourceDesc resourceDesc;
    memset(&resourceDesc, 0, sizeof(resourceDesc));
    resourceDesc.resType = cudaResourceTypeArray;

    /* Run our mandelbrot calculation kernel */
    const float centx = -0.5, centy = 0.0;
    const float diam = 2.5;
    float* c_count;
    cudaMalloc(&c_count, SCR_WIDTH * SCR_HEIGHT * sizeof(float));

    calculateMandelbrot(SCR_WIDTH, SCR_HEIGHT, MAX_ITERATIONS, centx, centy, diam, c_count);

    // Stuff to use for cuda interop
    cudaArray_t textureArray;
    cudaSurfaceObject_t surfaceObj = 0;

    // Tell opengl to slow down for our monitor's refresh rate
    glfwSwapInterval(1);

    while (!glfwWindowShouldClose(window))
    {
        // Process any user input
        processInput(window);

        // Map our mandelbrot to our created texture
        drawMandelbrot(SCR_WIDTH, SCR_HEIGHT, MAX_ITERATIONS, c_count, glfwGetTime(), &textureResource, &textureArray, &resourceDesc, surfaceObj);

        // Clear the color buffer from previous frame
        glClear(GL_COLOR_BUFFER_BIT);

        // Bind our texture
        glBindTexture(GL_TEXTURE_2D, texture);

        // Draw our texture, i.e. our mandelbrot
        ourShader.use();
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    /* Clean up */
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);

    cudaFree(c_count);

    glfwTerminate();
    return 0;
}

void processInput(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}
