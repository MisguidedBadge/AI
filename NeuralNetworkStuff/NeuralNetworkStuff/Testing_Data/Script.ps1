$directoryName = $null
$count = 0
Get-ChildItem "C:\Users\Peter\Desktop\AI\Repo\AI\NeuralNetworkStuff\NeuralNetworkStuff\Training_Data\Peds\" -File | 
  ForEach-Object {
   if ($directoryName -eq $null -or $directoryName -ne $_.DirectoryName) {
      #$directoryName = $_.DirectoryName
      #$count = 0
        $newname = "I" + $count++ + ".jpg"
        Write-Output $count
        Rename-Item $_.fullname $newname
  }

}