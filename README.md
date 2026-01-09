# Medical-Image-Processing



# Segmentation using traditional methods

**Autor: Kryštof Čejchan**
## Úvod
V tomto dokumentu popisuji, čím se první úkol zabýval, jak jsem tento úkol řešil, a jaké jsou výsledky.

## Popis projektu

První projekt se zabývá využitím monadických a morfologických operací nad bitmapovými obrázky, jejich binarizací, a interpretací.

<img alt="Tv21.tiff" height="510" src="segmentation%20-%20traditional%20methods/pics/popisprojektu.png" width="685"/>

Na vstupu je grey-scale bitmapový obrázek.

1. Obrázek projde pre-processingem, který odstraní nežádoucí objekty z obrázku, a zvýrazní žádoucí objekty v obrázku.
2. Obrázek se zbinarizuje, přičemž požedované objekty jsou v bílé barvě, zbytek je černý
3. Z původního grey-scale obrázku se vyříznou ty části, které jsou na binárním obrázku bílé
4. Tyto části projdou další vlnou operací, přičemž na konci je opět binární obrázek; avšak nyní se jedná už danou "buňku"

V nadcházejících kapitolách popíši přesné operace, které byly testovány nebo využity.
## Popis programu
Tato kapitola se zabývá, jak lze daný Python skript spustit a jak program využívat.

**Upozornění: program má grafické uživatelské rozhraní, proto je potřeba mít naistalovanou knihovnu tkinter (https://docs.python.org/3/library/tkinter.html)**
### Spuštění programu
Program se spouští pomocí `main.py` souboru, který otevře GUI.

![img.png](segmentation%20-%20traditional%20methods/pics/run_script.png)

Nyní je program "prázdný" a nelze využívat jeho funkcionality.
### Používání programu
Prvně musíme vybrat vstupní obrázek pomocí `File>Open...`.
Poté lze využívat operace, které jsou v horním menu.

- **File**: slouží k otevírání a ukládání obrázků, a ukončení programu.
- **Adjust**: obsahuje primárně operace z první přednášky; tj. monadické operace
- **Process**: obsahuje primárně operace z druhé přednášky; tj. morfologické operace
- **Filters&Blurs**: obsahuje primárně operace z třetí přednášky; tj. filtry a rozmazávání obrázku
- **Histograms**: slouží k zobrazování histogramů a CDF (cumulative distribution function), a nabízí také operace spojené s histogramem (ekvivalizace a CLAHE)
- **View**: slouží ke smazání provedených operací na obrázku
- **Pipelines**: obsahuje předvytvořené posloupnosti operací pro splnění cíle zadání

![img.png](segmentation%20-%20traditional%20methods/pics/ukazka_programu.png)

### Orientace ve zdrojovém kódu
Tato podkapitola se zabývá strukturou projektu z pohledu zdrojového kódu a dalších potřebných adresářů.

- `./main.py` je spouštěcí skript, volá třídu `ImageEditorApp`, která spustí GUI
- `./image_editor_ui.py` obsahuje třídu s GUI komponentami (tlačítka, textová pole, apod.), tato třída také ukládá vstupní a editovaný obrázek. Bohužel je soubor velice obsáhlý, protože GUI obsahuje mnoho komponent.
- `./image_operations/*.py` obsahuje naimplementované operace, které byly testovány nebo použity
  - `monadic_operations.py` obsahuje monadické operace, např. negace, kvantizace, nebo úpravu jasu, gammy, kontrastu
  - `advanced_operations.py` obsahuje "pokročilejší" operace jako např. CLAHE, erosion, dilatation, opening, closing, hledání kontur, nebo dokonce rozdělení spojených "buněk"
  - `filters.py` obsahuje operace filtrování a rozmazání, např.: gaussian blur, difference of gaussians, laplacian of gaussian...
  - `pipelines.py` obsahuje předvytvořené sekvence operací, které se spustí za účelem splnění cíle úkolu
- `./utils/*.py` obsahuje pomocné metody, např pro převod mězi Float ⟨0,1⟩ -> uint8 ⟨0,255⟩, a obsahuje fukce pro výpočet histogramu a CDF


## Řešení úkolu
Díky implementaci GUI jsem měl možnost si hrát s různými operacemi, testovat jejich parametry, a přitom ihned vidět výsledek.
Při řešení jsem implemetovat několik pipeline, ovšem nakonec jsem ponechal pouze jednu, která měla nejlepší výsledky.
Každá pipeline obsahuje dvě části:
1) operace nad vstupním obrázek
2) operace nad ROI (rectangle of interest)
### Pipeline 1
Tato pipeline nakonec vzešla jako ta nejlepší.
```python
def pipeline1(img: np.ndarray) -> np.ndarray:
    im = img.copy() 
    im = adjust_brightness(im, brightness=0.4) #pre-processing
    im = adjust_gamma(im, gamma=0.1)
    im = clahe(im, clip_limit=3, tile_grid=16)
    for _ in range(30):
        im = median_filter(im, ksize=3)
    im = negate(im) # binarizace
    im = otsu(im)
    im = opening(im, 10) # post-processing
    return im
```
Funkce nejprve aplikuje dvě základní operace: zvýší jas a sníží gammu. Tyto dvě operace, ačkoliv jsou jednoduché, efektivně 
zajistí, že na obrázku zůstanou pouze ty nejvýraznější objekty, tj. v podstatě se smaže nežádaný šum na pozadí; viz. obrázek níže

![img.png](segmentation%20-%20traditional%20methods/pics/brightgamma1.png)

Po použití CLAHE a median filtrů se vyhladí další drobný šum. (Median filter je v cyklu, protože OpenCV nedovoluje nastavit kernel size větší než 4)
Následuje negace a OTSU, které zajistí binarizaci.

![img.png](segmentation%20-%20traditional%20methods/pics/clahe_otsu1.png)

Pro odstranění dalších nežádoucích objektů se použije opening, který tyto objekty "ořeže" natolik, že při hledání contour, se nezaznamenají.

![img.png](segmentation%20-%20traditional%20methods/pics/opening1.png)

Výsledkem tohoto je množina obdélníkových obrázků grey-scale buňek. Tyto obrázky projdou další fází: binarizací a hledáním jader v buňce.
```python
def roi_binarization1(img: np.ndarray) -> np.ndarray:
    im = img.copy()
    im = erosion(im, erosion_size=1, interations_no=3)
    im = gaussian_blur(im, ksize=5, sigma=0)
    im = negate(im)
    im = adjust_gamma(im, gamma=0.16)
    im = otsu(im)
    im = opening(im, size=4, shape=cv.MORPH_RECT, iterations_no=1)
    return im
```
Nejprve se spustí erosion, která zvýrazný a zvětší černé části obrázku; v našem případě se jedná o jádra buňky.

![img.png](segmentation%20-%20traditional%20methods/pics/cellerso1.png)

Dále se aplikuje gaussian filter, který decentně rozmaže černé tečky, které vznikly po erosion.
Po negaci obrázku se sníží gamma (kdyby se obrázek neznegoval, gamma by se dala zvýšit pro podobný efekt). Výsledkem úpravy gammy je
obrázek se zvýrazněnými bílými buňkami.

![img.png](segmentation%20-%20traditional%20methods/pics/cellgamma1.png)

Po použití OTSU, je obrázek dokonán (alespoň v tomto případě); obecně to ale platit nemusí, proto ještě
použijeme opening. Opening se zde používá s tvarem `cv.MORPH_RECT`, protože bylo zjištěno, že obdélníkový tvar lépe
odděluje jádra, která jsou blízko u sebe, než elipsa.

![img.png](segmentation%20-%20traditional%20methods/pics/cellotsu.png)
![img_1.png](segmentation%20-%20traditional%20methods/pics/cell_open1.png)

Vzhledem k tomu, že u drtivé většiny obrázků, se prostřední dvě jádra spojí do jednoho, byla zvolena metoda, která do prostředního jádra
vloží černou úsečku. Viz. kód níže.
```python
def split_middle_cell(img, line_thickness=2):
    _, contours = find_contours(img, 30) #najde jádra
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0  

    best = None
    best_abs_d = float("inf")

    for c in contours: # pomocí pointPolygonTest, najde tu konturu, která je nejblíže středu obrázku
        d = cv.pointPolygonTest(c, (cx, cy), True)
        if abs(d) < best_abs_d:
            best_abs_d = abs(d)
            best = c

    middle_contour = best

    x, y, bw, bh = cv.boundingRect(middle_contour)
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(w, x + bw)
    y1 = min(h, y + bh)

    y_line = int((y0 + y1) / 2)
    cv.line(img, (x0, y_line), (x1, y_line), (0, 0, 0), thickness=line_thickness) #do středu "namaluje" černou úsečku

    return img
```

Po celé pipeline se ještě musí očíslovat dané kontoury buněk. Celá pipeline a její výsledek pak vypadají takto:
```python
def apply_pipeline(self, pipeline_number: int, contour_rect_size:int, binarization_method:int, contour_bin_cell_size:int=150):
        fun = getattr(pipelines, f'pipeline{pipeline_number}')# vzhledej pipeline podle jména funkce
        self.edited = fun(self.original) # spusť pipeline na vstupním obrázu; výsledek ulož do upraveného obrázku
        self.extract_rects_current(contour_rect_size,10) #pomocí obdélníkových kontour najdi "buňky"
        self.last_rois_binary.clear() 
        for i in range(len(self.last_rois) - 1, -1, -1):#pro každou najitou buňku...
            roi = self.last_rois[i]
            fun_bin = getattr(pipelines, f'roi_binarization{binarization_method}')
            binarized_roi = fun_bin(roi) #...spusť binarizaci buňky
            binarized_roi = split_middle_cell(binarized_roi) #...rozděl prostřední buňku
            contours = self.find_contours(binarized_roi, contour_bin_cell_size) #..pomocí kontour najdi "jádra buňky"
             #odstraň vše, co není v konturách
            mask = np.zeros(binarized_roi.shape[:2], dtype=np.uint8)
            cv.drawContours(mask, contours, -1, 1, thickness=cv.FILLED)
            binarized_roi[mask == 0] = 0
            if not (len(contours)==11): #pokud je buňka vadná, tj. neobsahuje 11 jader, vymaž ji
                self.last_rois.pop(i)
                continue
            counting = 0
            for c in contours:
                # očíseluj jádra a do pravého horního rohu napiš celkový počet jader
                x, y, bw, bh = cv.boundingRect(c)
                h, w = binarized_roi.shape[:2]

                x0 = max(0, x)
                y0 = max(0, y)
                x1 = min(w - 1, x + bw)
                y1 = min(h - 1, y + bh)

                cx = int(x0 + (x1 - x0) / 2)
                cy = int(y0 + (y1 - y0) / 2)
                
                counting +=1

                cv.putText(binarized_roi, str(counting), (cx, cy), cv.FONT_HERSHEY_PLAIN,0.5, 0,1)
            cv.putText(binarized_roi, str(counting), (5, 5), cv.FONT_HERSHEY_PLAIN,0.5, 1,1)
            self.last_rois_binary.append(binarized_roi)
        self.save_rects_current() #otevře dialog pro vybrání cílového adresáře a uloží tam output (binarizované a grey-scale obrázky  = dataset)
```

![binary_roi_006.png](segmentation%20-%20traditional%20methods/pics/pipeline_1/img1/binary_roi_006.png)

### Přehled naimplementovaných operací
Základní **jednoobrázkové (monadické)** operace – mění vlastnosti jasu, kontrastu a intenzity pixelů.

| Metoda | Popis | Využití |
|--------|--------|----------|
| **`negate(img)`** | Inverzuje obraz (vymění světlé a tmavé oblasti). | Zvýraznění hran |
| **`adjust_gamma(img, gamma)`** | Provádí korekci gama — upravuje jas obrazu nelineárně. | Kompenzace rozdílů v osvětlení, zjasnění nebo ztmavení snímků. |
| **`adjust_brightness(img, brightness)`** | Lineárně přidává nebo ubírá jas. | Korekce podexponovaných nebo přeexponovaných obrazů. |
| **`adjust_contrast(img, contrast)`** | Mění kontrast vynásobením pixelových hodnot. | Zvýraznění detailů, zlepšení ostrosti. |
| **`non_linear_contrast(img, alpha)`** | Nelineární úprava kontrastu pomocí přechodu mezi světly a stíny. | Přirozenější zlepšení kontrastu než lineární metoda. |
| **`logarithmic_scale(img, s)`** | Logaritmická transformace jasu. | Zvýraznění detailů ve tmavých oblastech. |
| **`quantization(img, q)`** | Snížení počtu úrovní jasu (kvantizace). | Zmenšení velikosti obrazu, stylizace, předzpracování pro klasifikaci. |

---
Kolekce **filtračních a hranových metod** – slouží k odstranění šumu, rozostření nebo zvýraznění struktur.

| Metoda | Popis | Využití |
|--------|--------|----------|
| **`mean_blur(img, ksize)`** | Průměrovací filtr. | Zmírnění šumu, vyhlazení obrazu. |
| **`gaussian_blur(img, ksize, sigma)`** | Gaussovské rozostření. | Potlačení šumu, odstranění detailů před segmentací. |
| **`median_filter(img, ksize)`** | Mediánový filtr. | Odstranění impulsního („salt & pepper“) šumu bez rozmazání hran. |
| **`bilateral_filter(img, d, sigma_color, sigma_space)`** | Bilaterální filtr – zachovává hrany při vyhlazování. | Redukce šumu při zachování kontur buněk nebo objektů. |
| **`canny(img, threshold1, _, threshold2)`** | Canny detektor hran. | Detekce hran a tvarů buněk, objektů nebo struktur. |
| **`diff_of_gaus(img, sigma, k)`** | Rozdíl dvou Gaussů (DoG). | Zvýraznění hran podobně jako Laplace, často v biologickém zpracování. |
| **`laplacian_of_gaus(img, sigma, ksize_lap)`** | Laplace filtr po Gaussovském rozostření (LoG). | Detekce hran a přechodů intenzity s menším šumem. |

---
Pokročilé **morfologické a segmentační** metody – užitečné pro analýzu buněk, objektů a tvarů.

| Metoda                                     | Popis                                                   | Využití                                                     |
|--------------------------------------------|---------------------------------------------------------|-------------------------------------------------------------|
| **`hist_equalization(img)`**               | Normalizace histogramu – vyrovná jas a kontrast.        | Zlepšení viditelnosti detailů.                              |
| **`clahe(img, clip_limit, tile_grid)`**    | Adaptivní ekvalizace histogramu.                        | Vyrovnání kontrastu u nehomogenně osvětlených snímků.       |
| **`erosion(img, size)`**                   | Eroze – zmenší světlé oblasti.                          | Odstranění drobných šumů nebo oddělení blízkých objektů.    |
| **`dilatation(img, size)`**                | Dilatace – rozšíří světlé oblasti.                      | Vyplnění mezer, zvýraznění struktur.                        |
| **`opening(img, size)`**                   | Otevření (eroze + dilatace).                            | Odstranění malých šumových bodů při zachování tvarů.        |
| **`closing(img, size)`**                   | Uzavření (dilatace + eroze).                            | Vyhlazení hran, odstranění malých tmavých děr.              |
| **`otsu(img)`**                            | Otsuův algoritmus pro automatickou prahovou segmentaci. | Automatické rozdělení obrazu na pozadí a objekty.           |
| **`find_contours(img, min_area)`**         | Najde obrysy (kontury) objektů.                         | Extrakce tvarů buněk nebo částic.                           |
| **`save_rect_images(rect_imgs, out_dir)`** | Uloží oříznuté oblasti jako samostatné snímky.          | Export jednotlivých objektů pro dataset.                    |
| **`split_middle_cell(img)`**               | Najde a rozdělí centrální buňku čarou.                  | Pomoc při separaci spojených buněk.                         |
| **`image_reconstruct(marker, img)`**       | Morfologická rekonstrukce.                              | Obnovení objektů z poškozených nebo částečně smazaných dat. |

---
## Výsledky
V následujicí tabulce popisuji, jak ve výsledku dopadl můj projekt.

| Název obrázku     | Počet "buněk" | Počet vadných buněk | Procento vadných | Konečný počet buněk |
|-------------------|---------------|---------------------|------------------|---------------------|
| PCD1.tiff         | 69            | 3                   | 4.34%            | 66                  |
| PCD2.tiff         | 18            | 7                   | 38.88%           | 11                  |
| PCD3.tiff         | 50            | 2                   | 4.00%            | 48                  |
| Tv8.tiff          | 18            | 0                   | 0.00%            | 18                  |
| Tv11.tiff         | 66            | 5                   | 8.12%            | 61                  |
| Tv17.tiff         | 24            | 3                   | 12.50%           | 21                  |
| Tv21.tiff         | 29            | 1                   | 3.44%            | 28                  |
| Tv31.tiff         | 30            | 5                   | 16.66%           | 25                  |
| Tv33.tiff         | 37            | 0                   | 0.00%            | 37                  |
| **SOUČET/PRŮMĚR** | **341**       | **26**              | **7.63%**        | **315**             |



# CNN implementation

Cílem tohoto úkolu bylo využít dataset z prvního úkolu a natrénovat na něm síť U-Net.

![test_00006.png](CNN%20implementation/out/test_predictions_25-11-04_14-14-44/test_preds/test_00006.png)

## Part 1: Dataset Preparation

Dataset se skládal ze zkruba 300 obrázků z prvního úkolu,
přičemž došlo ke změně, a to sice že se odstranila černá úsečka rozdělující prostřední tobuli cílie.
Dataset byl rozdělen na originální obrázky cílií a jejich černobílé masky.

Originální bitmapové obrázky byly načteny jako grey-scale a zvětšeny na rozlišení 256x256 pixelů pomocí bilineární interpolace.
Stejně byly načteny i masky, které ovšem byly zvětšeny pomocí interpolace nejbližího souseda, což v tomto případě šlo pro vést relativně bezproblémově, neboť se jedná o černobílé obrázky.

Pro zvětšení velikosti vstupního datasetu byla použita augmentace v podobě rotace obrázů o 90°, 180° a 270°. Díky takto zvoleným rotacím se nemuselo řešit, co se stane s prázným místem, kdybychom obrázek otočili o např. 45°. Také bylo zjištěno, že při zvýšení počtu rotací (např. kdybychom rotovali obrázek o 15°), tak by trénování sítě zabralo příliš mnoho času (za předpokladu, že bychom nenastavili počet epoch na nějaký nízký počet).

## Part 2: U-Net Architecture

U-Net je realizován v pětivrstvé architektuře (5 downsample bloků následovaných bottleneckem a 5 upsample bloky). Počet konvolučních filtrů na vstupní vrstvě je 16; v každém dalším stupni encoderu se počet filtrů zdvojnásobuje (tzn. 16, 32, 64, 128, 256). Zvhledem k tomu, že obrázek je na vstupu 256x256, tak se po poolováních v enkóderu dostane na 16x16.
Decoder je symetrický k encoderu a v jednotlivých úrovních postupně snižuje počet kanálů zpět na 16, přičemž využívá skip connection spojující odpovídající vrstvy.

Při implementaci výsledného řešení byly otestovány i některé úpravy, které ovšem nefungovaly dostatečně efektivně a proto nebyly použity. Mezi těmito úpravami bylo: 1） snížení počtu vrstev na 2, to způsobilo, že síť označovala některé části jako false positive.

![test_00005fp.png](CNN%20implementation/out/test_predictions_25-11-04_07-12-49/test_preds/test_00005.png)

2） Počet konvolučních filtrů na vstupní vrstvě byl nastaven na 32. Toto vedlo ke zvýšení výpočetní složitosti při trénování, aniž by se dostavily nějaké změny na efektivnosti výsledné sítě. Proto se zvolilo počátečních 16.

## Part 3: Model Training

Trénování modelu využívalo vytvořenou U-Netu, rozdělení datasetu na tři části, kombinaci dvou loss funkcí a early-stopping.

### Rozdělení datasetu

Dataset se rozdělil na tři části: trénovací, validační a testovací. Poměr těchto skupin by 70/15/15. Tento počet byl zvolen vzhledem k velikosti datasetu, jeho augmentaci a také protože to bylo zmíněno na přednášce.

```python
train_size = int(0.7 * len(ds)) # 70%
val_size   = int(0.15 * len(ds)) # 15%
test_size  = len(ds) - train_size - val_size # 100-70-15 = 15%
```

### Batch size

Při trénování byl také zvolen batch size 8 primárně kvůli mé velikosti paměti, každopádně podle mého neformálního výzkumu, by zvýšení batch size na 16 či 32 nemělo výrazný efekt.

### Ztrátová funkce

Při trénování modelu byly vyzkoušeny dvě loss funkce, a to sice dice loss, binary cross-entropy loss a následně i jejich kombinace. Tato kombinace využívala obou funkcí zároveň, přičemž jejich poměr byl určován hodnotou alpha, která určovala "váhu" BCE. Např. pokud alpha=0.8, tak BCE mělo váhu 80% a dice pouze 20%; pokud by alpha=0.5, funkce by měly stejnou váhu.

Nakonec se nejvíce osvědčilo použít buď BCE nebo dice; jejich kombinace měla úspěch pouze pokud byla alpha velmi nízká.

### Early stop a počet epoch

Počet epoch a early-stop spolu úzce souvisí, proto je popíši v rámci jedné podkapitoly.

Počet epoch byl nastaven na 120, ovšem kvůli early-stop algoritmu se trénovací algoritmus dostal nejvýše na 30.

Early-stop algoritmus po každé epoše zkontroluje validation loss; pokud se "zlepšila", tak algoritmus pokračuje dále, pokud se 5krát po sobě nezlepšila, trénování končí a dojde k obnovení nejlepší validační ztráty.
Za zlepšení se považuje následující přepis: `val_loss < best_val_loss - 1e-5`

### Visualizations of the process

V této podkapitole lze najít grafy různých metrik, které byly měřeny během trénovaní sítě s různými parametry. Mezi těmito paramatry většinou najdeme ztrátovou funkci, neboť early-stop a počet vrtev sítě byl nastaven staticky a k jeho změnám nedocházelo často.

Při zvolení ztrátové funkce BCE+Dice s váhou na BCE 0.9, trénování skončilo relativně rychle (po 12 epochách), ale hodnoty ztát dobré nebyly, viz. obrázek níže.

![bcedice09.png](CNN%20implementation/out/test_predictions_25-11-06_08-50-55_bce09/loss_curves.png)
![bcedice09.png](CNN%20implementation/out/test_predictions_25-11-06_08-50-55_bce09/dice_curve.png)

Opakem bylo zvolení BCE s 0.1, což způsobilo, že se síť trénovala 59 epoch než narazila na early-stop, ovšem její výsledky byly v porovnání s předchozím pokusem přívětivější.

![bcedice01.png](CNN%20implementation/out/test_predictions_25-11-06_09-05-23/loss_curves.png)
![bcedice01.png](CNN%20implementation/out/test_predictions_25-11-06_09-05-23/dice_curve.png)

Při zvolení BCE 0.5 (tj, obě ztrátové funkce měly stejnou váhu), tak výsledek byl opět lepší. Dosáhlo se toho, že se síť trénovala +-30 epoch se ztrátou < 0.1

Při zvolení pouze BCE ztrátové funkce, se dosáhlo nejlepších výsledků v porovnání s ostaními pokusy. Počet epoch byl pouze 32, přičemž ztráty a dice vykazovaly lepší hodnoty než předešlé pokusy.

![bcedice05.png](CNN%20implementation/out/test_predictions_25-11-06_10-23-06/loss_curves.png)
![bcedice05.png](CNN%20implementation/out/test_predictions_25-11-06_10-23-06/dice_curve.png)

## Part 4: Model Evaluation

### Qualitative

Kvalitativní evaluace modelu se provádí relativně těžce, protože výsledky se ukazují být lepší než ground truth. Toto je způsobeno, že v prvním úkolu se na tobule použila morfologická operace opening s obdélníkovým kernelem; výsledky sítě nejsou tedy tolik hranaté a působí "lépe".

Zvolená síť nakonec nenesla významné kvantitativní výkyvy. Tyto výkyvy (např. false positive) byly popsány v předchozích kapitolách, a byly způsobeny "mělkou" sítí nebo špatnou kombinací ztrátové funkce.

### Quantitative

V této kapitole jsou zobrazeny kvantitativní evaluace nejlepší sítě.

![bcedice05.png](CNN%20implementation/out/test_predictions_25-11-06_10-23-06/prf1_curves.png)

![bcedice05.png](CNN%20implementation/out/test_predictions_25-11-06_10-23-06/iou_curve.png)

![bcedice05.png](CNN%20implementation/out/test_predictions_25-11-06_10-23-06/accuracy_curve.png)

![bcedice05.png](CNN%20implementation/out/test_predictions_25-11-06_10-23-06/conf_matrix_epoch_37.png)

![bcedice05.png](<./out/test_predictions_25-11-06_10-23-06/progress_contours1(Copy).png>)
![bcedice05.png](<./out/test_predictions_25-11-06_10-23-06/progress_contours(Copy).png>)

## Part 5: Hyperparameter Tuning

V projektu jsem použil U-Net, což je konvoluční encoder–decoder architektura pro segmentaci obrazu. Síť se skládá ze dvou částí:

- encoder (downsampling): postupně zmenšuje rozlišení a zvyšuje počet filtrů, čímž extrahuje abstraktnější rysy z obrazu,

- decoder (upsampling): obnovuje prostorové rozlišení, přičemž využívá skip-connections z encoderu, takže kombinujeme nízkoúrovňové detaily s vysokoúrovňovou semantikou.

Každá úroveň obsahuje 2× konvoluci 3×3 a aktivaci ReLU. Mezi úrovněmi se používá MaxPooling (2×2) pro zmenšení rozlišení a transposed convolution (2×2) pro zvětšení rozlišení v decoderu. Výstupní vrstva je 1×1 konvoluce, která produkuje mapu logitů (1 kanál), vhodnou pro binární segmentaci pomocí BCEWithLogitsLoss.

Při použítí U-Net s 2 vrstvami došlo ke zvýšení počtu pixelů jako false positive; při použítí 10 vrstev byl trénink zbytečně dlouhý.

Jako nejlepší vyšla varianta pěti úrovní:

- poskytuje dostatečnou reprezentaci i pro jemné detaily,

- nemá tak velký počet parametrů, aby se přetrénovala,

- má nejlepší poměr výkon / doba tréninku / paměť,

- stabilní trénování a nejvyšší validační Dice a IoU.

Average pooling měl problém se zachováním hran a textur; max pooling v tomto vyšel jako lepší možnost.

## Ukázky výsledků

![vysledek](./out/test_predictions_25-11-06_10-23-06/test_preds/test_00076.png)

![vysledek](./out/test_predictions_25-11-06_10-23-06/test_preds/test_00040.png)

![vysledek](./out/test_predictions_25-11-06_10-23-06/test_preds/test_00077.png)

![vysledek](./out/test_predictions_25-11-06_10-23-06/test_preds/test_00028.png)





# Assingment 02

Cílem tohoto úkolu bylo využít dataset z prvního úkolu a natrénovat na něm síť U-Net.

![test_00006.png](CNN%20implementation/out/test_predictions_25-11-04_14-14-44/test_preds/test_00006.png)

## Part 1: Dataset Preparation

Dataset se skládal ze zkruba 300 obrázků z prvního úkolu,
přičemž došlo ke změně, a to sice že se odstranila černá úsečka rozdělující prostřední tobuli cílie.
Dataset byl rozdělen na originální obrázky cílií a jejich černobílé masky.

Originální bitmapové obrázky byly načteny jako grey-scale a zvětšeny na rozlišení 256x256 pixelů pomocí bilineární interpolace.
Stejně byly načteny i masky, které ovšem byly zvětšeny pomocí interpolace nejbližího souseda, což v tomto případě šlo pro vést relativně bezproblémově, neboť se jedná o černobílé obrázky.

Pro zvětšení velikosti vstupního datasetu byla použita augmentace v podobě rotace obrázů o 90°, 180° a 270°. Díky takto zvoleným rotacím se nemuselo řešit, co se stane s prázným místem, kdybychom obrázek otočili o např. 45°. Také bylo zjištěno, že při zvýšení počtu rotací (např. kdybychom rotovali obrázek o 15°), tak by trénování sítě zabralo příliš mnoho času (za předpokladu, že bychom nenastavili počet epoch na nějaký nízký počet).

## Part 2: U-Net Architecture

U-Net je realizován v pětivrstvé architektuře (5 downsample bloků následovaných bottleneckem a 5 upsample bloky). Počet konvolučních filtrů na vstupní vrstvě je 16; v každém dalším stupni encoderu se počet filtrů zdvojnásobuje (tzn. 16, 32, 64, 128, 256). Zvhledem k tomu, že obrázek je na vstupu 256x256, tak se po poolováních v enkóderu dostane na 16x16.
Decoder je symetrický k encoderu a v jednotlivých úrovních postupně snižuje počet kanálů zpět na 16, přičemž využívá skip connection spojující odpovídající vrstvy.

Při implementaci výsledného řešení byly otestovány i některé úpravy, které ovšem nefungovaly dostatečně efektivně a proto nebyly použity. Mezi těmito úpravami bylo: 1） snížení počtu vrstev na 2, to způsobilo, že síť označovala některé části jako false positive.

![test_00005fp.png](CNN%20implementation/out/test_predictions_25-11-04_07-12-49/test_preds/test_00005.png)

2） Počet konvolučních filtrů na vstupní vrstvě byl nastaven na 32. Toto vedlo ke zvýšení výpočetní složitosti při trénování, aniž by se dostavily nějaké změny na efektivnosti výsledné sítě. Proto se zvolilo počátečních 16.

## Part 3: Model Training

Trénování modelu využívalo vytvořenou U-Netu, rozdělení datasetu na tři části, kombinaci dvou loss funkcí a early-stopping.

### Rozdělení datasetu

Dataset se rozdělil na tři části: trénovací, validační a testovací. Poměr těchto skupin by 70/15/15. Tento počet byl zvolen vzhledem k velikosti datasetu, jeho augmentaci a také protože to bylo zmíněno na přednášce.

```python
train_size = int(0.7 * len(ds)) # 70%
val_size   = int(0.15 * len(ds)) # 15%
test_size  = len(ds) - train_size - val_size # 100-70-15 = 15%
```

### Batch size

Při trénování byl také zvolen batch size 8 primárně kvůli mé velikosti paměti, každopádně podle mého neformálního výzkumu, by zvýšení batch size na 16 či 32 nemělo výrazný efekt.

### Ztrátová funkce

Při trénování modelu byly vyzkoušeny dvě loss funkce, a to sice dice loss, binary cross-entropy loss a následně i jejich kombinace. Tato kombinace využívala obou funkcí zároveň, přičemž jejich poměr byl určován hodnotou alpha, která určovala "váhu" BCE. Např. pokud alpha=0.8, tak BCE mělo váhu 80% a dice pouze 20%; pokud by alpha=0.5, funkce by měly stejnou váhu.

Nakonec se nejvíce osvědčilo použít buď BCE nebo dice; jejich kombinace měla úspěch pouze pokud byla alpha velmi nízká.

### Early stop a počet epoch

Počet epoch a early-stop spolu úzce souvisí, proto je popíši v rámci jedné podkapitoly.

Počet epoch byl nastaven na 120, ovšem kvůli early-stop algoritmu se trénovací algoritmus dostal nejvýše na 30.

Early-stop algoritmus po každé epoše zkontroluje validation loss; pokud se "zlepšila", tak algoritmus pokračuje dále, pokud se 5krát po sobě nezlepšila, trénování končí a dojde k obnovení nejlepší validační ztráty.
Za zlepšení se považuje následující přepis: `val_loss < best_val_loss - 1e-5`

### Visualizations of the process

V této podkapitole lze najít grafy různých metrik, které byly měřeny během trénovaní sítě s různými parametry. Mezi těmito paramatry většinou najdeme ztrátovou funkci, neboť early-stop a počet vrtev sítě byl nastaven staticky a k jeho změnám nedocházelo často.

Při zvolení ztrátové funkce BCE+Dice s váhou na BCE 0.9, trénování skončilo relativně rychle (po 12 epochách), ale hodnoty ztát dobré nebyly, viz. obrázek níže.

![bcedice09.png](CNN%20implementation/out/test_predictions_25-11-06_08-50-55_bce09/loss_curves.png)
![bcedice09.png](CNN%20implementation/out/test_predictions_25-11-06_08-50-55_bce09/dice_curve.png)

Opakem bylo zvolení BCE s 0.1, což způsobilo, že se síť trénovala 59 epoch než narazila na early-stop, ovšem její výsledky byly v porovnání s předchozím pokusem přívětivější.

![bcedice01.png](CNN%20implementation/out/test_predictions_25-11-06_09-05-23/loss_curves.png)
![bcedice01.png](CNN%20implementation/out/test_predictions_25-11-06_09-05-23/dice_curve.png)

Při zvolení BCE 0.5 (tj, obě ztrátové funkce měly stejnou váhu), tak výsledek byl opět lepší. Dosáhlo se toho, že se síť trénovala +-30 epoch se ztrátou < 0.1

Při zvolení pouze BCE ztrátové funkce, se dosáhlo nejlepších výsledků v porovnání s ostaními pokusy. Počet epoch byl pouze 32, přičemž ztráty a dice vykazovaly lepší hodnoty než předešlé pokusy.

![bcedice05.png](CNN%20implementation/out/test_predictions_25-11-06_10-23-06/loss_curves.png)
![bcedice05.png](CNN%20implementation/out/test_predictions_25-11-06_10-23-06/dice_curve.png)

## Part 4: Model Evaluation

### Qualitative

Kvalitativní evaluace modelu se provádí relativně těžce, protože výsledky se ukazují být lepší než ground truth. Toto je způsobeno, že v prvním úkolu se na tobule použila morfologická operace opening s obdélníkovým kernelem; výsledky sítě nejsou tedy tolik hranaté a působí "lépe".

Zvolená síť nakonec nenesla významné kvantitativní výkyvy. Tyto výkyvy (např. false positive) byly popsány v předchozích kapitolách, a byly způsobeny "mělkou" sítí nebo špatnou kombinací ztrátové funkce.

### Quantitative

V této kapitole jsou zobrazeny kvantitativní evaluace nejlepší sítě.

![bcedice05.png](CNN%20implementation/out/test_predictions_25-11-06_10-23-06/prf1_curves.png)

![bcedice05.png](CNN%20implementation/out/test_predictions_25-11-06_10-23-06/iou_curve.png)

![bcedice05.png](CNN%20implementation/out/test_predictions_25-11-06_10-23-06/accuracy_curve.png)

![bcedice05.png](CNN%20implementation/out/test_predictions_25-11-06_10-23-06/conf_matrix_epoch_37.png)

![bcedice05.png](./CNN%20implementation/out/test_predictions_25-11-06_10-23-06/progress_contours1(Copy).png)
![bcedice05.png](./CNN%20implementation/out/test_predictions_25-11-06_10-23-06/progress_contours(Copy).png)

## Part 5: Hyperparameter Tuning

V projektu jsem použil U-Net, což je konvoluční encoder–decoder architektura pro segmentaci obrazu. Síť se skládá ze dvou částí:

- encoder (downsampling): postupně zmenšuje rozlišení a zvyšuje počet filtrů, čímž extrahuje abstraktnější rysy z obrazu,

- decoder (upsampling): obnovuje prostorové rozlišení, přičemž využívá skip-connections z encoderu, takže kombinujeme nízkoúrovňové detaily s vysokoúrovňovou semantikou.

Každá úroveň obsahuje 2× konvoluci 3×3 a aktivaci ReLU. Mezi úrovněmi se používá MaxPooling (2×2) pro zmenšení rozlišení a transposed convolution (2×2) pro zvětšení rozlišení v decoderu. Výstupní vrstva je 1×1 konvoluce, která produkuje mapu logitů (1 kanál), vhodnou pro binární segmentaci pomocí BCEWithLogitsLoss.

Při použítí U-Net s 2 vrstvami došlo ke zvýšení počtu pixelů jako false positive; při použítí 10 vrstev byl trénink zbytečně dlouhý.

Jako nejlepší vyšla varianta pěti úrovní:

- poskytuje dostatečnou reprezentaci i pro jemné detaily,

- nemá tak velký počet parametrů, aby se přetrénovala,

- má nejlepší poměr výkon / doba tréninku / paměť,

- stabilní trénování a nejvyšší validační Dice a IoU.

Average pooling měl problém se zachováním hran a textur; max pooling v tomto vyšel jako lepší možnost.

## Ukázky výsledků

![vysledek](./CNN%20implementation/out/test_predictions_25-11-06_10-23-06/test_preds/test_00076.png)

![vysledek](./CNN%20implementation/out/test_predictions_25-11-06_10-23-06/test_preds/test_00040.png)

![vysledek](./CNN%20implementation/out/test_predictions_25-11-06_10-23-06/test_preds/test_00077.png)

![vysledek](./CNN%20implementation/out/test_predictions_25-11-06_10-23-06/test_preds/test_00028.png)



# Advanced CNN, transfer learning, and XAI

> Author: Kryštof Čejchan

## Objective
Implementace a evaluace klasifikačních modelů (CNN), analýza jejich rozhodování (XAI) a Siamské sítě.
## Part 1: Classification (Original vs. Inpainted)

### Příprava datasetu

Dataset byl připraven z bitmapových obrazů z předešlých úloh. Dohromady měl dataset ± 300 obrazů, z toho 50% bylo inpainted a zbytek byl ponechán beze změn (50%).

Pro inpainted byly použity masky cílií, které byly zdilatovány a následně pomocí funkce `cv.inpaint` bylo nejméně pět náhodných buněk "překresleno".

<img alt="img_008.png" height="256" src="Advanced%20CNN,%20transfer%20learning,%20and%20XAI/readme_files/img_008.png" width="256"/>
<img alt="7.png" height="256" src="Advanced%20CNN,%20transfer%20learning,%20and%20XAI/data/orig_inpainted/inpainted/7.png" width="256"/>

### Modely CNN

#### Vlastní síť

Feature Extractor: Série pěti konvolučních bloků, které postupně snižují prostorovou dimenzi obrazu a zvyšují hloubku příznaků (počet kanálů).

Klasifikátor: Plně propojené (Fully Connected) vrstvy, které převádějí extrahované příznaky na finální predikci.

Na vstupu je greyscale obrázek (tj. jeden kanál) s rozlišením 256x256. Další vrstvy jsou následujicí:


| Část sítě    | Vrstva | Typ Operace      | Konfigurace                  | Výstupní Tenzor |
|:-------------|:-------|:-----------------|:-----------------------------|:----------------|
| Vstup        | -      | -                | -                            | (1, 256, 256)   |
| Blok 1       | conv1  | Konvoluce + ReLU | k=5, s=1, p=2                | (16, 256, 256)  |
| -            | pool1  | Max Pooling      | k=2, s=2                     | (16, 128, 128)  |
| Blok 2       | conv2  | Konvoluce + ReLU | k=5, s=1, p=2                | (32, 128, 128)  |
| -            | pool2  | Max Pooling      | k=2, s=2                     | (32, 64, 64)    |
| Blok 3       | conv3  | Konvoluce + ReLU | k=3, s=1, p=1                | (64, 64, 64)    |
| -            | pool3  | Max Pooling      | k=2, s=2                     | (64, 32, 32)    |
| Blok 4       | conv4  | Konvoluce + ReLU | k=3, s=1, p=1                | (128, 32, 32)   |
| -            | pool4  | Max Pooling      | k=2, s=2                     | (128, 16, 16)   |
| Blok 5       | conv5  | Konvoluce + ReLU | k=3, s=1, p=1                | (256, 16, 16)   |
| -            | pool5  | Max Pooling      | k=2, s=2                     | (256, 8, 8)     |
| Flatten      | -      | Zploštění        | -                            | (256 * 64)      |
| Klasifikátor | fc1    | Linear + ReLU    | Vstup: 256 * 64, Výstup: 512 | (512)           |
| -            | fc2    | Linear + ReLU    | Vstup: 512, Výstup: 128      | (128)           |
| -            | fc3    | Linear (Logits)  | Vstup: 128, Výstup: 2        | (2)             |

`k = kernel_size; s = stride; p = padding`


#### Transfer learning

Standardní ResNet18 je navržen pro barevné RGB obrázky (3 kanály) a klasifikaci do 1000 tříd. Pro potřeby této úlohy (greyscale obrázky [1 kanál], klasifikace do 2 tříd) byly provedeny následující modifikace:

1. Adaptace vstupní vrstvy (Grayscale)

   Původní vstupní konvoluční vrstva (conv1) očekává 3 vstupní kanály (RGB). Jelikož pracujeme s černobílými (grayscale) obrázky (1 kanál), byla tato vrstva nahrazena novou konvolucí:

   Původní: in_channels=3
   Nová: in_channels=1 (ostatní parametry jako kernel size, stride a padding zůstaly zachovány).

   Aby se nepřišlo o naučené informace z RGB verze, váhy nové vrstvy nebyly inicializovány náhodně. Místo toho byl vypočítán průměr vah přes původní 3 kanály.

    ```py
    self.model.conv1.weight.data = original_conv1.weight.data.mean(dim=1, keepdim=True)
    ```

   Tato technika umožňuje síti reagovat na strukturní rysy v černobílém obraze podobně, jako by reagovala na jasovou složku barevného obrazu.

2. Úprava klasifikační hlavy (Head)

   Původní plně propojená vrstva (fc), která mapovala příznaky na 1000 tříd ImageNetu, byla odstraněna a nahrazena novou lineární vrstvou odpovídající našemu zadání:

   Vstup: 512 příznaků (výstup z posledního ResNet bloku).

   Výstup: num_classes (v našem případě 2: Original vs. Inpainted).

3. Strategie trénování (Freezing)

   Třída podporuje parametr freeze_base, který umožňuje zmrazit váhy extraktoru příznaků:

   Pokud freeze_base=True: Gradienty se počítají pouze pro novou klasifikační hlavu (fc). To je vhodné pro rychlé doladění (fine-tuning), kdy předpokládáme, že naučené příznaky z ImageNetu jsou dostatečně obecné.

   Pokud freeze_base=False (výchozí v kódu): Trénuje se celá síť. Váhy z ImageNetu slouží jako velmi kvalitní startovní bod inicializace, ale během tréninku se jemně přizpůsobují specifikům datasetu cílií.

### Trénink

1. Rozdělení a příprava dat

   Data jsou načítána z adresářové struktury, kde jsou třídy (original, inpainted) odděleny do podadresářů. Před samotným trénováním probíhá následující zpracování:

   Rozdělení datasetu: Všechny dostupné snímky jsou náhodně zamíchány a rozděleny na tři disjunktní sady na základě definovaných poměrů (70/15/15):

   Trénovací sada: Slouží k optimalizaci vah modelu.

   Validační sada: Slouží k průběžnému vyhodnocování modelu a rozhodování o předčasném ukončení (Early Stopping).

   Testovací sada: Použita výhradně po skončení trénování pro finální změření výkonnosti modelu.

   Data Augmentace: Pro zvýšení robustnosti modelu a prevenci přeučení (overfitting) je na trénovací sadu aplikována augmentace dat ve formě rotací o 90°, 180° a 270°. Validační a testovací sady zůstávají bez rotací (úhel 0°).

2. Konfigurace trénování

   Pro optimalizaci modelu byly zvoleny následující parametry a komponenty:

   Ztrátová funkce: Byla použita CrossEntropyLoss

   Optimalizátor: Byl zvolen algoritmus Adam (Adaptive Moment Estimation), který efektivně přizpůsobuje rychlost učení pro jednotlivé parametry sítě.

   Model: Skript využívá třídu ResNetClassifier nebo Net (podle toho, zda se jedná o transfer learning nebo ne), která je inicializována a přesunuta na výpočetní zařízení CPU.

3. Trénovací smyčka a Early Stopping

   Trénování probíhá v cyklech (epochách). Každá epocha se skládá ze dvou fází:

   Trénovací fáze (model.train()):

   Model zpracovává data po dávkách (batches).

   Pro každou dávku se vypočítá chyba (loss), provedou se zpětné propagace (backpropagation) a aktualizují se váhy pomocí optimalizátoru.

   Validační fáze (model.eval()):    
   Model je přepnut do evaluačního režimu (vypnutí dropoutu, fixace batch norm).

   Bez výpočtu gradientů (torch.no_grad()) se provede predikce na validační sadě.

   Výpočet metrik: Výstupy sítě (logity pro 2 třídy) jsou transformovány na binární predikci rozdílem skóre (class_1 - class_0), což umožňuje výpočet Accuracy, Precision, Recall a F1-Score pomocí třídy Metrics.

   Strategie Early Stopping (Předčasné zastavení): Aby se předešlo přeučení a plýtvání výpočetním časem, je implementován mechanismus Early Stopping.

   Monitoruje se hodnota validační ztráty (validation loss).

   Pokud je aktuální validační ztráta nižší než doposud nejlepší zaznamenaná, model (jeho váhy) se uloží jako nejlepší kandidát.

   Pokud se ztráta nezlepší po stanovený počet epoch (parametr PATIENCE), trénování je automaticky ukončeno.

4. Vizualizace a finální testování

   Po ukončení tréninku skript generuje grafy průběhu:

   Loss Graph: Porovnání vývoje trénovací a validační chyby v čase.

   Accuracy Graph: Vývoj přesnosti modelu na validační sadě.

   V poslední fázi se načtou váhy modelu s nejnižší validační chybou (nikoliv váhy z poslední epochy) a provede se inferenece na testovací sadě.

   Snímky jsou na základě predikce fyzicky roztříděny do složek predicted_original a predicted_inpainted pro vizuální kontrolu.

   Jsou vypočítány finální metriky výkonnosti na datech, která model během tréninku nikdy neviděl.

### Výstupy
Moje síť:

![accuracy_net.png](Advanced%20CNN,%20transfer%20learning,%20and%20XAI/readme_files/accuracy_net.png)

Transfer learning:

![accuracy_net.png](Advanced%20CNN,%20transfer%20learning,%20and%20XAI/readme_files/resnet/accuracy_net.png)

Moje síť:

![graph_loss_net.png](Advanced%20CNN,%20transfer%20learning,%20and%20XAI/readme_files/graph_loss_net.png)

Transfer learning:

![graph_loss_net.png](Advanced%20CNN,%20transfer%20learning,%20and%20XAI/readme_files/resnet/graph_loss_net.png)




Moje síť:

![confusion_matrix.png](Advanced%20CNN,%20transfer%20learning,%20and%20XAI/readme_files/confusion_matrix.png)

| Metrika   | Skóre  |
|-----------|--------|
| Precision | 0.7391 |
| Recall    | 1.0000 |
| F1-Score  | 0.8500 |
| IoU       | 0.7391 |


Transfer learning:

![confusion_matrix.png](Advanced%20CNN,%20transfer%20learning,%20and%20XAI/readme_files/resnet/confusion_matrix.png)


| Metrika   | Skóre  |
|-----------|--------|
| Precision | 1.0000 |
| Recall    | 1.0000 |
| F1-Score  | 1.0000 |
| IoU       | 1.0000 |


## Part 2: Model Interpretability

Pro validaci rozhodovacího procesu neuronové sítě a ověření, zda se model zaměřuje na relevantní vizuální znaky, byla implementována sada metod pro vysvětlitelnou umělou inteligenci (Explainable AI - XAI). K tomuto účelu byla využita knihovna Captum, která umožňuje analyzovat příspěvky jednotlivých pixelů k finální predikci modelu.

Analýza byla provedena na natrénovaném modelu (Net nebo ResNetClassifier) s využitím tří odlišných gradientních metod. Každá z nich poskytuje jiný pohled na to, co model považuje za důležité.

Použité metody vizualizace

1. Saliency (Gradient-based):

   Základní metoda, která počítá gradient výstupu vzhledem ke vstupnímu obrázku. Výsledná mapa indikuje, které pixely by při malé změně nejvíce ovlivnily výsledné skóre třídy.

   Vizualizace: Používá absolutní hodnotu gradientů (sign="absolute_value") a barevnou mapu inferno, aby zvýraznila oblasti s nejvyšší citlivostí bez ohledu na směr vlivu.


2. Integrated Gradients (IG):

   Tato metoda řeší problém saturace gradientů integrováním gradientů podél cesty od referenčního "nulového" vstupu (černý obrázek) k aktuálnímu vstupu. Poskytuje stabilnější a méně zašuměné výsledky než prostá Saliency mapa.

   Vizualizace: Zobrazuje pouze pozitivní příspěvky (sign="positive", mapa Reds), tedy ty oblasti, které přímo zvyšují pravděpodobnost predikované třídy.

3. Guided Grad-CAM:

   Kombinuje lokalizační schopnost metody Grad-CAM s detailním rozlišením Guided Backpropagation. Tato metoda sleduje aktivace v poslední konvoluční vrstvě sítě, která obsahuje nejvyšší úroveň sémantické informace.

   Konfigurace: Jako cílová vrstva (target_layer) byla pro vlastní síť Net zvolena vrstva conv5 (u ResNetu by to byl poslední blok layer4). Tato vrstva slouží jako zdroj pro výpočet vah důležitosti jednotlivých map příznaků.

   Vizualizace: Výstupem je detailní mapa (mapa viridis), která zvýrazňuje klíčové struktury (např. hrany nebo textury řasinek) vedoucí k rozhodnutí.

### Moje síť
![compare_18.png](Advanced%20CNN,%20transfer%20learning,%20and%20XAI/readme_files/compare_18.png)

![compare_98.png](Advanced%20CNN,%20transfer%20learning,%20and%20XAI/readme_files/compare_98.png)

![compare_99.png](Advanced%20CNN,%20transfer%20learning,%20and%20XAI/readme_files/compare_99.png)

![compare_102.png](Advanced%20CNN,%20transfer%20learning,%20and%20XAI/readme_files/compare_102.png)

### Resnet

![compare_0.png](Advanced%20CNN,%20transfer%20learning,%20and%20XAI/readme_files/resnet/compare_0.png)

![compare_1.png](Advanced%20CNN,%20transfer%20learning,%20and%20XAI/readme_files/resnet/compare_1.png)

![compare_10.png](Advanced%20CNN,%20transfer%20learning,%20and%20XAI/readme_files/resnet/compare_10.png)

![compare_55.png](Advanced%20CNN,%20transfer%20learning,%20and%20XAI/readme_files/resnet/compare_55.png)

![compare_62.png](Advanced%20CNN,%20transfer%20learning,%20and%20XAI/readme_files/resnet/compare_62.png)


### Porovnání (moje síť vs resnet)
![compare_0.png](Advanced%20CNN,%20transfer%20learning,%20and%20XAI/readme_files/compare_0.png)

![compare_0.png](Advanced%20CNN,%20transfer%20learning,%20and%20XAI/readme_files/resnet/compare_0.png)

![compare_99.png](Advanced%20CNN,%20transfer%20learning,%20and%20XAI/readme_files/compare_99.png)

![compare_99.png](Advanced%20CNN,%20transfer%20learning,%20and%20XAI/readme_files/resnet/compare_99.png)

## Part 3: Siamese Networks
Pro úlohu detekce inpaintingu byla implementována Siamská neuronová síť (Siamese Neural Network). Na rozdíl od klasických klasifikačních sítí, které se učí přiřadit vstupu konkrétní třídu, se siamská síť učí metriku podobnosti. Cílem je transformovat vstupní obrázky do vektorového prostoru (embedding space) tak, aby vektory obrázků stejné třídy byly blízko u sebe a vektory rozdílných tříd daleko od sebe.

1. Architektura modelu

   Jako základ (backbone) modelu byl zvolen ResNet18, předtrénovaný na ImageNetu, což zajišťuje robustní extrakci příznaků.

   Adaptace vstupu: Jelikož vstupní data jsou černobílá (1 kanál), byla první konvoluční vrstva ResNetu modifikována. Původní váhy pro 3 RGB kanály byly zprůměrovány do jednoho kanálu, což umožňuje využít předtrénované informace i pro grayscale vstupy.

   Sdílené váhy: Síť se skládá ze dvou identických větví, které sdílejí tytéž váhy. Oba obrázky z páru procházejí stejnou transformací.

   Embedding vrstva: Původní klasifikační hlava ResNetu byla nahrazena sekvencí Linear -> ReLU -> Linear, která mapuje extrahované příznaky do 128-dimenzionálního výstupního vektoru.

2. Příprava dat a párování

   Pro trénování siamské sítě je klíčová tvorba párů. Třída SiameseDataset generuje trénovací vzorky dynamicky:

   Pozitivní pár (Label 0): Dva různé obrázky stejné třídy (např. Original–Original nebo Inpainted–Inpainted).

   Negativní pár (Label 1): Dva obrázky rozdílných tříd (Original–Inpainted).

   Vyvážení: Dataset je konstruován tak, aby pravděpodobnost výběru pozitivního a negativního páru byla 50:50, což zabraňuje biasu sítě k jedné z variant.

3. Ztrátová funkce (Contrastive Loss)

   K optimalizaci vah byla použita funkce Contrastive Loss. Tato funkce pracuje s Euklidovskou vzdáleností Dw
   mezi výstupními vektory sítě.

4. Průběh trénování

   Trénování probíhá pomocí optimalizátoru Adam s learning rate 0.0005.

   Evaluace přesnosti: Přesnost modelu není měřena klasicky, ale na základě prahování vzdálenosti. Pokud je vzdálenost mezi vektory menší než threshold=margin/2, je pár klasifikován jako "shodný".

   Early Stopping: Pro zabránění přeučení je monitorována validační ztráta (Loss). Pokud se nezlepší po stanovený počet epoch (PATIENCE), trénování je předčasně ukončeno a uloží se model s nejnižší validační chybou.

### Metriky
#### VÝSLEDKY EVALUACE (Threshold=0.5)
| Metrika  | Skóre |
|----------|-------|
| Accuracy | 0.6   |
| Recall   | 0.72  |
| F1-Score | 0.51  |
| IoU      | 0.6   |

| Metrika  | Skóre |
|----------|-------|
| Accuracy | 0.6   |
| Recall   | 0.72  |
| F1-Score | 0.51  |
| IoU      | 0.6   |

| Třída (Class)    | Precision | Recall | F1-Score | Support |
|:-----------------|:---------:|:------:|:--------:|:-------:|
| **Stejné (0)**   |   0.51    |  0.72  |   0.60   |   25    |
| **Různé (1)**    |   0.72    |  0.51  |   0.60   |   35    |
|                  |           |        |          |         |
| **Accuracy**     |           |        | **0.60** | **60**  |
| **Macro Avg**    |   0.62    |  0.62  |   0.60   |   60    |
| **Weighted Avg** |   0.63    |  0.60  |   0.60   |   60    |


![training_metrics_siam.png](Advanced%20CNN,%20transfer%20learning,%20and%20XAI/readme_files/siam/training_metrics_siam.png)
![siamese_embeddings_vis.png](Advanced%20CNN,%20transfer%20learning,%20and%20XAI/readme_files/siam/siamese_embeddings_vis.png)
![siamese_confusion_matrix.png](Advanced%20CNN,%20transfer%20learning,%20and%20XAI/readme_files/siam/siamese_confusion_matrix.png)


# Movement detection & optical flow

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

![SLICO - Segmentation_screenshot_16.12.2025.png](Movement%20detection%20&%20optical%20flow/output/superpixels/3_new-run/SLICO%20-%20Segmentation_screenshot_16.12.2025.png)
![SLICO - Distance Heatmap_screenshot_16.12.2025.png](Movement%20detection%20&%20optical%20flow/output/superpixels/3_new-run/SLICO%20-%20Distance%20Heatmap_screenshot_16.12.2025.png)

### parametry: region_size=30, ruler=100.0

![SLICO - Segmentation_screenshot_16.12.2025.png](Movement%20detection%20&%20optical%20flow/output/superpixels/3_new_run_3/SLICO%20-%20Segmentation_screenshot_16.12.2025.png)
![SLICO - Distance Heatmap_screenshot_16.12.2025.png](Movement%20detection%20&%20optical%20flow/output/superpixels/3_new_run_3/SLICO%20-%20Distance%20Heatmap_screenshot_16.12.2025.png)

### parametry: region_size=20, ruler=8.0

![SLICO - Segmentation_screenshot_16.12.2025.png](Movement%20detection%20&%20optical%20flow/output/superpixels/3_new_run_2/SLICO%20-%20Segmentation_screenshot_16.12.2025.png)
![SLICO - Distance Heatmap_screenshot_16.12.2025.png](Movement%20detection%20&%20optical%20flow/output/superpixels/3_new_run_2/SLICO%20-%20Distance%20Heatmap_screenshot_16.12.2025.png)

## LSC

`region_size` opět měnil akorát velikost zaregistrovaných segmetací. `ratio` měnilo "zaoblení" segment, čím větší
hodnota, tím více "hranatě" segment vypadal.

### parametry: region_size=30, ratio=0.075

![LSC - Segmentation_screenshot_16.12.2025.png](Movement%20detection%20&%20optical%20flow/output/superpixels/3_new-run/LSC%20-%20Segmentation_screenshot_16.12.2025.png)
![LSC - Distance Heatmap_screenshot_16.12.2025.png](Movement%20detection%20&%20optical%20flow/output/superpixels/3_new-run/LSC%20-%20Distance%20Heatmap_screenshot_16.12.2025.png)

### parametry: region_size=20, ratio=0.5

![LSC - Segmentation_screenshot_16.12.2025.png](Movement%20detection%20&%20optical%20flow/output/superpixels/3_new_run_2/LSC%20-%20Segmentation_screenshot_16.12.2025.png)
![LSC - Distance Heatmap_screenshot_16.12.2025.png](Movement%20detection%20&%20optical%20flow/output/superpixels/3_new_run_2/LSC%20-%20Distance%20Heatmap_screenshot_16.12.2025.png)

## SEEDS

U SEEDS metody je zajímavé si všimnout heatmapy, kde je vidět, že byly zvoleny superpixely na základě `num_superpixels`,
což mnohdy vede k "prázdným" superpixelům.

### parametry: num_superpixels=8000, num_levels=20

![SEEDS - Segmentation_screenshot_16.12.2025.png](Movement%20detection%20&%20optical%20flow/output/superpixels/3_new-run/SEEDS%20-%20Segmentation_screenshot_16.12.2025.png)
![SEEDS - Distance Heatmap_screenshot_16.12.2025.png](Movement%20detection%20&%20optical%20flow/output/superpixels/3_new-run/SEEDS%20-%20Distance%20Heatmap_screenshot_16.12.2025.png)

### parametry: num_superpixels=6000, num_levels=10

![SEEDS - Segmentation_screenshot_16.12.2025.png](Movement%20detection%20&%20optical%20flow/output/superpixels/3_new_run_2/SEEDS%20-%20Segmentation_screenshot_16.12.2025.png)
![SEEDS - Distance Heatmap_screenshot_16.12.2025.png](Movement%20detection%20&%20optical%20flow/output/superpixels/3_new_run_2/SEEDS%20-%20Distance%20Heatmap_screenshot_16.12.2025.png)

# Part 2: Visual comparison of background subtraction methods.

Tato část se věnuje analýze pohybu ve videosekvencích pomocí metod background subtraction a optického toku.

## Background Subtraction

Výstupem je čistá maska pohybujících se objektů, která slouží jako vstup pro další analýzu nebo počítání objektů.

![Screencast from 2025-12-13 13-31-04.gif](https://krystofcejchan.cz/projects/medical_img_processing/Screencast%20from%202025-12-13%2013-31-04.gif)

# Sparse Optical Flow

Nejprve jsou v obraze detekovány "význačné body" pomocí detektoru `goodFeaturesToTrack`. Tyto body
typicky odpovídají rohům nebo výrazným texturám na vozidlech.

Metoda následně vypočítává posun těchto bodů mezi po sobě jdoucími snímky.

Výsledkem je vizualizace pohybových vektorů, které ukazují dráhu pohybu jednotlivých objektů v čase. Narorozdíl od
`dense optical flow` byla tato metoda výpočeetně nenáročná a zvládala video zpracovávat v realném čase.

![Screencast from 2025-12-13 13-40-29.gif](https://krystofcejchan.cz/projects/medical_img_processing/Screencast%20from%202025-12-13%2013-40-29.gif)

![Screencastfrom2025-12-1315-33-59-ezgif.com-video-to-gif-converter.gif](https://krystofcejchan.cz/projects/medical_img_processing/Screencastfrom2025-12-1315-33-59-ezgif.com-video-to-gif-converter.gif)

# Dense Optical Flow

Další implemetovaná metoda byla Dense Optical Flow, která vypočítává vektor pohybu
pro každý pixel v obraze - výpočet byl náročný, proto se zvolilo video s nízkým rozlišením.

Výstupem je husté pole vektorů, které popisuje pohyb celé scény.

Pro vizualizaci byl použit barevný kód (HSV prostor), kde barva určuje směr pohybu (např. červená = doprava, modrá =
doleva) a sytost/jas určuje rychlost pohybu (jasnější = rychlejší pohyb).

## Dense optiocal flow bez úprav morfologickými operacemi

![Screencastfrom2025-12-1313-50-54-ezgif.com-video-to-gif-converter.gif](https://krystofcejchan.cz/projects/medical_img_processing/Screencastfrom2025-12-1313-50-54-ezgif.com-video-to-gif-converter.gif)

## Dense optiocal flow se upravami morfologickými operacemi

Při použití morfologické operace `close`, se lépe vyplnila maska, což způsobilo, že se bounding rectangle nezmenšoval (
viz. předchozí obrázek)

![Screencastfrom2025-12-1314-54-53-ezgif.com-video-to-gif-converter.gif](Movement%20detection%20&%20optical%20flow/output/dense/Screencastfrom2025-12-1314-54-53-ezgif.com-video-to-gif-converter.gif)

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

Neřeší se zde identita objektů, pouze dektekce pohybu. Při kolizi se jejich pohybové masky spojí do jedné velké kontury.
Na rozdíl od sparse metody zde není mechanismus, který by poznal, že se "ztratil" konkrétní bod. Objekt je prostě po
dobu kolize vnímán jako jeden velký pohybující se shluk.
