import qbs 1.0
import '../QtModule.qbs' as QtModule

QtModule {
    qtModuleName: "EglFSDeviceIntegration"
    Depends { name: "Qt"; submodules: ["core", "gui", "core-private", "gui-private", "devicediscovery_support-private", "eventdispatcher_support-private", "service_support-private", "theme_support-private", "fontdatabase_support-private", "fb_support-private", "egl_support-private", "input_support-private", "platformcompositor_support-private"]}

    architecture: "x86_64"
    hasLibrary: true
    staticLibsDebug: []
    staticLibsRelease: []
    dynamicLibsDebug: []
    dynamicLibsRelease: ["", "gthread-2.0", "glib-2.0", "", "", "/usr/lib/x86_64-linux-gnu/libQt5DBus.so.5.9.5", "", "fontconfig", "freetype", "", "", "Xext", "X11", "m", "EGL", "", "mtdev", "input", "xkbcommon", "", "GL", "/usr/lib/x86_64-linux-gnu/libQt5Gui.so.5.9.5", "", "udev", "/usr/lib/x86_64-linux-gnu/libQt5Core.so.5.9.5", "pthread", "dl"]
    linkerFlagsDebug: []
    linkerFlagsRelease: []
    frameworksDebug: []
    frameworksRelease: []
    frameworkPathsDebug: []
    frameworkPathsRelease: []
    libNameForLinkerDebug: "Qt5EglFSDeviceIntegration"
    libNameForLinkerRelease: "Qt5EglFSDeviceIntegration"
    libFilePathDebug: ""
    libFilePathRelease: "/usr/lib/x86_64-linux-gnu/libQt5EglFSDeviceIntegration.so.5.9.5"
    cpp.defines: ["QT_EGLFSDEVICEINTEGRATION_LIB"]
    cpp.includePaths: ["/usr/include/x86_64-linux-gnu/qt5", "/usr/include/x86_64-linux-gnu/qt5/QtEglFSDeviceIntegration", "/usr/include/x86_64-linux-gnu/qt5/QtEglFSDeviceIntegration/5.9.5", "/usr/include/x86_64-linux-gnu/qt5/QtEglFSDeviceIntegration/5.9.5/QtEglFSDeviceIntegration"]
    cpp.libraryPaths: []
    
}
