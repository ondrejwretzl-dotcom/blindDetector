@ECHO OFF
set DIR=%~dp0
"%JAVA_HOME%\bin\java.exe" -jar "%DIR%gradle\wrapper\gradle-wrapper.jar" %*
