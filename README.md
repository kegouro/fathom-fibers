# Fathom Fibers Quick

MVP local para medir ancho proyectado de fibras en micrografías SEM. Está optimizado para TIFF Zeiss con metadata `CZ_SEM`, pero también abre TIFF/PNG/JPEG genéricos con calibración manual.

## Qué funciona ahora

- verificación e integridad de imagen fuente por SHA-256 (`MATCH`, `MISSING`, `MISMATCH`, `UNVERIFIED`);
- validación y reconstrucción geométrica de `width_m` al cargar proyectos;
- gates geométricos de validación para propuestas manuales y asistidas;
- resumen principal por fibra (`fiber-level summary` basado en medianas por fibra) y secundario por secciones locales;
- protección contra pérdida de cambios no guardados (dirty state);
- lectura automática de `ap_image_pixel_size` en TIFF Zeiss;
- detección y exclusión visual del footer Zeiss;
- visor con zoom y paneo;
- medición manual con dos clics;
- ajuste de bordes (snap asistido) y propuesta local de un clic con puntajes heurísticos;
- edición de extremos arrastrándolos;
- clasificación de fibras en 1–4 grupos usando la mediana por fibra;
- guardado de proyecto JSON con coordenadas, hash y provenance básico;
- exportación CSV, PNG anotado e informe HTML;
- CLI para inspeccionar TIFF y crear inventarios de carpetas.

## Qué no afirma todavía

No segmenta automáticamente toda una red densa ni reconstruye fibras ocultas bajo cruces. La propuesta de un clic entrega un puntaje heurístico local (basado en anisotropía de gradientes y perfil de bordes), no una probabilidad calibrada. No se afirma validación científica externa ni oráculo automático. En redes superpuestas, el conteo mostrado corresponde a las IDs de fibra que el operador creó o confirmó.

## Ejecución y verificación rápida

Ejecutar tests desde la raíz:

```bash
python -m pytest -q
```

Ejecutar suite completa de calidad:

```bash
./scripts/check.sh
```

## Instalación rápida en Arch Linux

```bash
sudo pacman -S --needed python tk
./install.sh
./run.sh
```

El instalador crea `.venv` e instala las dependencias del proyecto.

También puedes hacerlo manualmente:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m fathom_fibers_quick gui
```

## Protocolo recomendado

1. Abre el TIFF y comprueba la escala mostrada en el panel izquierdo.
2. Usa una ID nueva para cada fibra, por ejemplo `F001`.
3. Realiza entre 3 y 5 secciones limpias y aproximadamente perpendiculares por fibra.
4. Evita cruces, bordes de imagen y regiones fusionadas. Si son relevantes, márcalos como defecto.
5. Para máxima confianza usa **Manual 2 clics**.
6. **Ajustar bordes** toma tus dos clics como aproximación y busca gradientes cercanos.
7. **Propuesta local 1 clic** estima orientación y bordes. Siempre revísala con la herramienta `V` y arrastra los extremos.
8. Clasifica únicamente después de medir varias fibras. Los grupos son descriptivos y no prueban por sí solos familias físicas.
9. Guarda el proyecto y exporta CSV + informe HTML.

## Controles

- rueda: zoom;
- botón derecho y arrastrar: paneo;
- `V`: seleccionar y editar;
- `M`: manual, dos clics;
- `S`: snap asistido, dos clics;
- `A`: propuesta local, un clic;
- `Delete`: eliminar seleccionada;
- `Esc`: cancelar medición pendiente;
- `Ctrl+O`: abrir imagen;
- `Ctrl+S`: guardar proyecto.

## CLI

Inspeccionar una imagen:

```bash
python -m fathom_fibers_quick inspect "PVDF Jose_01.tif"
```

Crear inventario de una carpeta:

```bash
python -m fathom_fibers_quick inventory ./imagenes -o zeiss_inventory.csv
```

## Estructura científica

Cada medición conserva:

- extremos subpíxel;
- ancho físico en metros (recalculado y verificado geométricamente);
- método;
- fibra asociada;
- confianza heurística local, cuando aplica;
- grupo;
- defecto/observación;
- aceptación;
- hash del archivo y calibración en el proyecto/exportación.

La magnitud fundamental es **ancho proyectado en la micrografía 2D**. Interpretarlo como diámetro requiere asumir una fibra aproximadamente cilíndrica, visible, no fusionada y cercana al plano de imagen.
