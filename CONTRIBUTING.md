# Contribuir al proyecto Traductor

¡Gracias por tu interés en contribuir! Este documento describe cómo colaborar con el proyecto.

## Código de conducta

Este proyecto se adhiere al [Código de Conducta](CODE_OF_CONDUCT.md). Al participar, se espera que respetes este código.

## Cómo contribuir

### Reportar bugs

1. Verificá que el bug no haya sido reportado ya en [Issues](https://github.com/whispem/sussurro.cpp/issues)
2. Usá la plantilla de **Bug Report** para crear un nuevo issue
3. Incluí toda la información necesaria para reproducir el problema

### Sugerir funcionalidades

1. Verificá que la funcionalidad no haya sido sugerida ya
2. Usá la plantilla de **Feature Request**
3. Describí el problema que resuelve y la solución propuesta

### Enviar código (Pull Requests)

1. Fork del repositorio
2. Creá una rama desde `main`: `git checkout -b feature/mi-funcionalidad`
3. Hacé tus cambios siguiendo las convenciones del proyecto
4. Commiteá con mensajes descriptivos (conventional commits)
5. Push a tu fork: `git push origin feature/mi-funcionalidad`
6. Abrí un Pull Request

## Convenciones de código

### Commits

Usar [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: agregar voice cloning con Qwen3-TTS
fix: corregir path hardcodeado en lib.rs
docs: actualizar SETUP.md
refactor: extraer VAD a módulo separado
```

### Código C++

- Estándar C++17
- Seguir el estilo existente del proyecto
- Nombres de variables en snake_case
- Clases en PascalCase

### Código Rust (Tauri)

- Seguir las convenciones de Rust (rustfmt)
- Usar `cargo fmt` antes de commitear
- Manejo de errores con `Result<T, String>`

### Código JavaScript/HTML/CSS

- Indentación de 2 espacios
- Sin frameworks externos (vanilla JS)

## Estructura del proyecto

```
traductor/
├── engine/          # C++ (ggml) — traducción Marian
├── app/             # Tauri v2 (Rust + JS)
├── models/          # Modelos de IA (no commitear)
├── scripts/         # Scripts de setup
└── third_party/     # Submodules (ggml, whisper.cpp, miniaudio)
```

## Build local

```bash
# Inicializar submodules
git submodule update --init --recursive

# Compilar engine
cmake -B build && cmake --build build -j

# Compilar app
cd app && npm install && npm run tauri dev
```

## Licencia

Al contribuir, aceptás que tus contribuciones se licencien bajo la licencia MIT del proyecto.
