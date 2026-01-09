# Assignment 4

> Autor: Kryštof Čejchan


_Pozn.: v readme jsou gify, některé se načítají delší dobu kvůli jejich velikosti._

V rámci tohoto úkolu jsem implementoval algoritmy a metody pro superpixely a detekci poh ybu (motion detection).

# Part 1: Comparison of superpixel methods and parameter choices (number of superpixels, threshold values).

Tato část úkolu se zaměřovala na segmentaci snímků pomocí superpixelových metod. Cílem bylo nahradit práci s
jednotlivými pixely efektivnější reprezentací pomocí superpixelů a klasifikovat je na základě barevné podobnosti s
referenčním vzorkem.

Prvním krokem bylo definování cílové barvy, kterou chceme v obrázku detekovat.
Pro tento účel je nejprve zobrazen vstupní obrázek uživateli, který vybere kurzorem roi. Barvy pixelů v tomto roi se
zprůměrují, což je výsledná cílová barva.

```python
roi_rect = cv.selectROI("select rectangle sample", img, showCrosshair=True, fromCenter=False)
```

Při testování se volila následující barva jako ta cílová: `(L*a*b*): [ 60.25 146.97 122.47]`

Extrakce superpixelů byla implementovaná pomocí tří metod: SLICO, LSC, SEEDS. Threshold pro segmentaci byl nastaven 85.

## SLICO

Byly vyzkoušeny dvě nastavení parametrů algoritmů, jejich efekt se nijak zásadně nelišil. Při `region_size=20` je vidět,
že jsou regiony menší (logicky), ale jinak je segmentace téměř identická; heatmapa každopádně působí více přesněji bez
velkých regionů, ale bylo to za cenu výpočetní složitosti.
`ruler` parametr se na výsledném obrázku moc neprojevil, kromě pár nových segmentů.

### parametry: region_size=30, ruler=15.0

![SLICO - Segmentation_screenshot_16.12.2025.png](output/superpixels/3_new-run/SLICO%20-%20Segmentation_screenshot_16.12.2025.png)
![SLICO - Distance Heatmap_screenshot_16.12.2025.png](output/superpixels/3_new-run/SLICO%20-%20Distance%20Heatmap_screenshot_16.12.2025.png)

### parametry: region_size=30, ruler=100.0

![SLICO - Segmentation_screenshot_16.12.2025.png](output/superpixels/3_new_run_3/SLICO%20-%20Segmentation_screenshot_16.12.2025.png)
![SLICO - Distance Heatmap_screenshot_16.12.2025.png](output/superpixels/3_new_run_3/SLICO%20-%20Distance%20Heatmap_screenshot_16.12.2025.png)

### parametry: region_size=20, ruler=8.0

![SLICO - Segmentation_screenshot_16.12.2025.png](output/superpixels/3_new_run_2/SLICO%20-%20Segmentation_screenshot_16.12.2025.png)
![SLICO - Distance Heatmap_screenshot_16.12.2025.png](output/superpixels/3_new_run_2/SLICO%20-%20Distance%20Heatmap_screenshot_16.12.2025.png)

## LSC

`region_size` opět měnil akorát velikost zaregistrovaných segmetací. `ratio` měnilo "zaoblení" segment, čím větší
hodnota, tím více "hranatě" segment vypadal.

### parametry: region_size=30, ratio=0.075

![LSC - Segmentation_screenshot_16.12.2025.png](output/superpixels/3_new-run/LSC%20-%20Segmentation_screenshot_16.12.2025.png)
![LSC - Distance Heatmap_screenshot_16.12.2025.png](output/superpixels/3_new-run/LSC%20-%20Distance%20Heatmap_screenshot_16.12.2025.png)

### parametry: region_size=20, ratio=0.5

![LSC - Segmentation_screenshot_16.12.2025.png](output/superpixels/3_new_run_2/LSC%20-%20Segmentation_screenshot_16.12.2025.png)
![LSC - Distance Heatmap_screenshot_16.12.2025.png](output/superpixels/3_new_run_2/LSC%20-%20Distance%20Heatmap_screenshot_16.12.2025.png)

## SEEDS

U SEEDS metody je zajímavé si všimnout heatmapy, kde je vidět, že byly zvoleny superpixely na základě `num_superpixels`,
což mnohdy vede k "prázdným" superpixelům.

### parametry: num_superpixels=8000, num_levels=20

![SEEDS - Segmentation_screenshot_16.12.2025.png](output/superpixels/3_new-run/SEEDS%20-%20Segmentation_screenshot_16.12.2025.png)
![SEEDS - Distance Heatmap_screenshot_16.12.2025.png](output/superpixels/3_new-run/SEEDS%20-%20Distance%20Heatmap_screenshot_16.12.2025.png)

### parametry: num_superpixels=6000, num_levels=10

![SEEDS - Segmentation_screenshot_16.12.2025.png](output/superpixels/3_new_run_2/SEEDS%20-%20Segmentation_screenshot_16.12.2025.png)
![SEEDS - Distance Heatmap_screenshot_16.12.2025.png](output/superpixels/3_new_run_2/SEEDS%20-%20Distance%20Heatmap_screenshot_16.12.2025.png)

# Part 2: Visual comparison of background subtraction methods.

Tato část se věnuje analýze pohybu ve videosekvencích pomocí metod background subtraction a optického toku.

## Background Subtraction

Výstupem je čistá maska pohybujících se objektů, která slouží jako vstup pro další analýzu nebo počítání objektů.

![Screencast from 2025-12-13 13-31-04.gif](output/bcg/Screencast%20from%202025-12-13%2013-31-04.gif)

# Sparse Optical Flow

Nejprve jsou v obraze detekovány "význačné body" pomocí detektoru `goodFeaturesToTrack`. Tyto body
typicky odpovídají rohům nebo výrazným texturám na vozidlech.

Metoda následně vypočítává posun těchto bodů mezi po sobě jdoucími snímky.

Výsledkem je vizualizace pohybových vektorů, které ukazují dráhu pohybu jednotlivých objektů v čase. Narorozdíl od
`dense optical flow` byla tato metoda výpočeetně nenáročná a zvládala video zpracovávat v realném čase.

![Screencast from 2025-12-13 13-40-29.gif](output/sparse/Screencast%20from%202025-12-13%2013-40-29.gif)

![Screencastfrom2025-12-1315-33-59-ezgif.com-video-to-gif-converter.gif](output/sparse/Screencastfrom2025-12-1315-33-59-ezgif.com-video-to-gif-converter.gif)

# Dense Optical Flow

Další implemetovaná metoda byla Dense Optical Flow, která vypočítává vektor pohybu
pro každý pixel v obraze - výpočet byl náročný, proto se zvolilo video s nízkým rozlišením.

Výstupem je husté pole vektorů, které popisuje pohyb celé scény.

Pro vizualizaci byl použit barevný kód (HSV prostor), kde barva určuje směr pohybu (např. červená = doprava, modrá =
doleva) a sytost/jas určuje rychlost pohybu (jasnější = rychlejší pohyb).

## Dense optiocal flow bez úprav morfologickými operacemi

![Screencastfrom2025-12-1313-50-54-ezgif.com-video-to-gif-converter.gif](output/dense/Screencastfrom2025-12-1313-50-54-ezgif.com-video-to-gif-converter.gif)

## Dense optiocal flow se upravami morfologickými operacemi

Při použití morfologické operace `close`, se lépe vyplnila maska, což způsobilo, že se bounding rectangle nezmenšoval (
viz. předchozí obrázek)

![Screencastfrom2025-12-1314-54-53-ezgif.com-video-to-gif-converter.gif](output/dense/Screencastfrom2025-12-1314-54-53-ezgif.com-video-to-gif-converter.gif)

# Part 2 (Discussion): Answers to the implementation challenges in Optical Flow (handling new objects, duplicates, and collisions).

## Detekce nových objektů

### Sparse

V každém třetím snímku (`frame_idx % detect_interval == 0`) se volá funkce `cv.goodFeaturesToTrack`. Tím se v obraze
najdou nové výrazné body, které se přidají do seznamu tracks.

### Dense

Funkce `cv.calcOpticalFlowFarneback` počítá vektory pohybu pro celý obraz v každém snímku. Následně se aplikuje
prahování (`cv.threshold`) na velikost vektorů.

## Řešení duplicit

### Sparse

Před zavoláním detektoru nových bodů se vytvoří maska, kde jsou pozice aktuálně sledovaných bodů začerněny (
`hodnota 0`). Funkce `cv.goodFeaturesToTrack` hledá body pouze v bílých oblastech masky. Tím je matematicky zaručeno, že
se nezačne sledovat bod, který už systém eviduje.

```jupyter
mask = np.zeros_like(frame_gray)
mask[:] = 255
for tr in tracks:
    x, y = tr[-1]
    cv.circle(mask, (int(x), int(y)), 5, 0, -1)
```

### Dense

Dense optical flow může být zašuměný a jeden objekt může vypadat jako mnoho malých skvrn, tj. jeden objekt by byl
zaregostrován vícekrát. Morfologická operace close spojí tyto blízké body do jednoho celistvého blobu, čímž zamezí
vícenásobné detekci jednoho auta.

```jupyter
processed_mask = cv.morphologyEx(motion_mask, cv.MORPH_CLOSE, k, iterations=9)
```

## Řešení kolizí

### Sparse

Algoritmus spočítá pohyb bodu z času t do t+1 a hned poté zpětný pohyb z t+1 do t.
Pokud je sledování správné, měl by se bod vrátit přesně na své původní místo. Pokud došlo ke kolizi nebo zákrytu, bod
se "ztratí" nebo se přichytí na jiný objekt, a zpětný pohyb skončí jinde. Pokud je vzdálenost d větší než 1 pixel, bod
je vyřazen jako nespolehlivý.

```jupyter
p1, _, _ = cv.calcOpticalFlowPyrLK(img0, img1, p0, ...)
p0r, _, _ = cv.calcOpticalFlowPyrLK(img1, img0, p1, ...)
d = abs(p0 - p0r).reshape(-1, 2).max(-1)
good = d < 1
```

### Dense

Neřeší se yde identita objektů, pouze dektekce pohybu. Při kolizi se jejich pohybové masky spojí do jedné velké kontury.
Na rozdíl od sparse metody zde není mechanismus, který by poznal, že se "ztratil" konkrétní bod. Objekt je prostě po
dobu kolize vnímán jako jeden velký pohybující se shluk.
