for dir in ~/Documents/radiohead/*/; do 
    for f in ${dir}*.mp3; do 
        echo $f;
        filename=$(basename "$f");
        directory=$(basename "$dir");
        mkdir -p radiohead_downsampled/${directory}
        sox "$f" -r 16000 -c 1 "radiohead_downsampled/${directory}/${filename}"
    done; 
done;