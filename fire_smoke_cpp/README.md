# Smoke & Flame C++ v1.2

Natiivi C++20-porttaus käyttäjän Python/ModernGL-version **v0.8.2** fluidimallista ja GLSL-varjostimesta. 240×135-laskentaruudukko päivitetään CPU:lla; savun ja liekin renderöinti tehdään OpenGL 3.3:ssa neljällä R32F-tekstuurilla.

## Arch Linux

```bash
sudo pacman -S --needed base-devel cmake sdl2 sdl2_ttf glew
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/fire_smoke
```

Ohjelma etsii shaderit suoritettavan tiedoston viereisestä `shaders/`-hakemistosta. Käynnistä projektin juuresta tai `build/`-hakemistosta.

## Näppäimet

- `Esc` lopeta, `Space` tauko, `R` nollaa, `F` tuli päälle/pois
- `T` lämpötilan debug, `G` polttoaineen debug
- `L` volumetrinen savu päälle/pois, `[` ja `]` säätävät voimakkuutta
- Hiiren vasen painike lisää polttoainetta ja lämpöä osoittimen kohdalle
- `+` / `-` säädä tulen ja savun lähteen voimakkuutta ajon aikana
- `O` kytke kirjainten fluidieste päälle/pois (teksti pysyy näkyvissä)

## Muutokset v0.8.2:een

- Lähtöliekki on noin 1.6× leveämpi ja tehokkaampi; korkeus syntyy suuremmasta lämmöstä ja nousuvoimasta.
- Palamisesta vapautuu noin 2.4× enemmän savua ja liekin renderöintimaski laajenee 3–4 ruutua.
- Sama paineprojektio, vorticity confinement, bilineaarinen advektio, punainen liekkipaletti ja savun volumetrinen lisäkerros kuin Python-versiossa.
- Korjattu shaderin reunasämpläys clampaamalla liekkimaski simulaation sisään.
- Keskellä Montserratilla **PROMETHEUS** ja sen alla kapealla Cormorant Garamond -antiikvalla **The Usurper of Fire**.
- Tekstin glyyfit toimivat fluidisimulaatiossa esteinä: nouseva savu kiertää kirjaimet ja pääsee väleistä. `O` vertaa estevaikutusta.
- v1.2: liekin lähdealue kolminkertainen v1.1:een nähden (x-suuntainen säde 12,8 → 38,4), savun elinikä noin 20 % pidempi (hajoamiskerroin 0,996 → 0,996664 per päivitys).

Montserrat ja Cormorant Garamond sisältyvät `assets/`-hakemistoon Google Fontsin SIL Open Font License -lisensseillä (`assets/OFL.txt` ja `assets/OFL-Cormorant.txt`).

Tämä on 2D-fluididemo; alkuperäinen v0.8.2 ei sisällä kuution 3D-mallia. Toteutus ei lisää sellaista omatoimisesti.
