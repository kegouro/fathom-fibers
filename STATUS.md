# Estado del entregable 0.1.0 — MVP Endurecido

## Verificado en este paquete

- Verificación Headless de Integridad de la fuente del proyecto (`MATCH`, `MISSING`, `MISMATCH`, `UNVERIFIED`) mediante SHA-256.
- Reconstrucción y validación geométrica de `width_m` al cargar proyectos para evitar confiar ciegamente en valores serializados.
- Gates geométricos para mediciones asistidas y manuales (coordenadas finitas, límites de imagen, footer excluido, mínimo 2 px, límite del dominio de búsqueda).
- Resumen principal por fibra (`fiber-level summary`) para evitar sesgos por fibras supermedidas, junto con el resumen de secciones locales.
- Mecanismo de dirty state en GUI con confirmación de descarte de cambios y marcador `*` en el título.
- Script de verificación unificado (`scripts/check.sh`) ejecutando pytest, compileall, ruff y git diff --check.
- Ejecución directa de `python -m pytest -q` desde la raíz sin requerir `PYTHONPATH=src`.
- Suite automatizada ampliada a 11 tests focales aprobados.

## Capacidades terminadas

- medición manual con validación de límites;
- snap clásico de bordes;
- propuesta local de un clic con puntajes heurísticos;
- edición interactiva de extremos;
- verificación e integridad de SHA-256 de imágenes;
- recálculo y corrección de anchos derivados;
- IDs de fibra y múltiples secciones;
- estadísticas por fibra (medianas) y por secciones locales;
- grupos de tamaño por mediana de fibra;
- marcado manual de defectos;
- protección contra pérdida de cambios no guardados;
- proyecto JSON;
- CSV, PNG e HTML;
- CLI de metadata e inventario;
- contratos y entry points para backends futuros.

## Límites conocidos

- No existe aún segmentación automática global de todas las fibras ni modelos ML.
- Las puntuaciones de herramientas asistidas son heurísticas locales (anisotropía + gradientes), no probabilidades calibradas ni oráculos.
- No se afirma validación científica externa.
- La clasificación es descriptiva y no demuestra familias materiales por sí sola.
- El conteo corresponde a IDs creadas/revisadas por el operador.
- El ancho es proyectado en 2D. Interpretarlo como diámetro requiere asumir geometría aproximadamente cilíndrica.
- Undo/redo, autosave y trazado continuo de centerline se reservan para fases futuras.

## Recomendación de uso inmediato

Para datos que deban defenderse científicamente, usar medición manual de 3–5 secciones limpias por fibra. Usar el snap o la propuesta local para acelerar, pero revisar y ajustar siempre los extremos antes de aceptar la medición.
