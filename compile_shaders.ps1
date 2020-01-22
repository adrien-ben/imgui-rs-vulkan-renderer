Get-ChildItem -Path .\src\shaders\ -File -Recurse -exclude *.spv | ForEach-Object { 
    $sourcePath = $_.fullname
    $targetPath = "$($_.fullname).spv"
    glslangValidator -V -o $targetPath $sourcePath
}
