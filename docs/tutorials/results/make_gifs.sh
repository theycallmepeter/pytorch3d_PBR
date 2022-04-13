for d in */*/; do
    convert -delay 20 -loop 0 "$d"prediction_*.* "$d"prediction.gif;
    convert -delay 20 -loop 0 "$d"novel_view_prediction_*.* "$d"novel_view.gif;
    convert -delay 20 -loop 0 "$d"silhouette_prediction_*.* "$d"silhouette.gif;
    convert -delay 20 -loop 0 "$d"texture_*.* "$d"texture.gif;
    convert -delay 20 -loop 0 "$d"normalmap_*.* "$d"normalmap.gif;
    echo $d;
done