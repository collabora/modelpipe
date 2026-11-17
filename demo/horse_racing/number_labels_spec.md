# Horse number labels

Each sheet `NNNN.png` has 30 numbered crops of horses from Venezuelan race broadcasts. `NNNN.csv` maps each cell number to a crop name. Give each cell one class: the saddle cloth number of the horse the box is around, `1` to `14`, or `none`.

The saddle cloth is the coloured pad under the saddle, behind the jockey's leg, with the number printed on it. Jockey silks (shirt, cap) and hoods or blinkers on the horse's head are owner colours, not the number. Ignore them.

Cloth colours:

| number | cloth | printed number |
| --- | --- | --- |
| 1 | red | white |
| 2 | white | black |
| 3 | blue | white |
| 4 | yellow | black |
| 5 | green | white |
| 6 | black | yellow |
| 7 | orange | black |
| 8 | pink | black |
| 9 | turquoise | red |
| 10 | purple | white |
| 11 | grey | red |
| 12 | lime | black |
| 13 | brown | white |
| 14 | maroon | yellow |

1 to 12 are seen on this footage, though the number on the turquoise 9 cloth looked dark blue rather than red. 13 and 14 are the common US colours and are not confirmed. Under the night lights yellow 4 looks gold and orange 7 looks red-orange, and pink 8 can look pale.

Rules:

- A readable printed number decides the class, whatever the colour looks like. Note it when the number and colour table disagree.
- Without a readable number, the cloth colour decides it, when one cloth colour is clearly visible.
- The horse is the one the box is centred on and that fills most of the box. If two horses fill the box about equally, use `none`.
- `none`: the main horse's cloth is hidden, too blurred or too small to tell the colour, the colour fits two classes (gold 4 or orange 7, red 1 or orange 7) with no readable number, or the box is not a horse.
- Head-on and tail-on views show the cloth as strips at the sides of the horse. Label them when the colour is clear.

Write `DATA/number_labels/<batch>.csv` with the header `sheet,number,class,note`, one row per filled cell, `sheet` as the four digit name without extension. `note` is empty unless a printed number disagreed with the colour table or something else on the sheet looked wrong. The last sheet may have fewer than 30 filled cells, the `.csv` lists which.
