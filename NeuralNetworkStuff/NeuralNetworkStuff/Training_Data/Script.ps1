$directoryName = $null
$count = 100
Get-ChildItem "C:\Users\Peter\Desktop\AI\Repo\AI\NeuralNetworkStuff\NeuralNetworkStuff\Testing_Data\NonPedTest" -File | 
  ForEach-Object {
   if ($directoryName -eq $null -or $directoryName -ne $_.DirectoryName) {
      #$directoryName = $_.DirectoryName
      #$count = 0
        $newname = "I" + $count++ + ".jpg"
        Write-Output $count
        Rename-Item $_.fullname $newname
  }

}