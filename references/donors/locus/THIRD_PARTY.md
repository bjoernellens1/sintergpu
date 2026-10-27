# Third-party dependencies

Original project code is under MIT. External dependencies are resolved at build
time and retain their own licenses and terms. No vendor AARs or fonts are
bundled. Inspect upstream notices before distributing an APK.

| Dependency | Declared version / source | Role |
|---|---|---|
| Android Gradle Plugin | 8.10.1 | Android build |
| Gradle Wrapper | 8.11.1, Apache 2.0 | Bundled wrapper JAR and launch scripts |
| Kotlin / Compose compiler | 2.1.20 | Application and core |
| Compose BOM | 2025.05.01 | Material 3 UI |
| ARCore | 1.56.0 | Camera tracking/depth |
| AndroidX Media3 | 1.11.1, Apache 2.0 | Local MP4 replay (ExoPlayer and UI) |
| USB serial for Android | 3.9.0 | Serial transport |
| RealSense | v2.56.5 | Optional external camera wrapper |
| Orbbec Android | commit 8822458b10e51e5c2818b22ab2f549aaab575da1 | Optional v2.0.6 AAR |
| MCAP format | mcap.dev specification | Original native writer implementation |

API references are listed in docs/SOURCES.md. CI actions and external Python
oracles are declared in the workflow and are not bundled implementations.
