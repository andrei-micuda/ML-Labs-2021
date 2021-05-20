`CL32C7V~CL32C7V~BN~P2~DO50 -> CL64C5V~CL64C5V~BN~P2~DO50 -> F -> D128~DO50 -> D128~DO50 -> D3`

Cu droput optimizat de `hyperas`:

`CL32C7V~CL32C7V~BN~P2~DO18 -> CL64C5V~CL64C5V~BN~P2~DO82 -> F -> D128~DO26 -> D128~DO62 -> D3`. Antrenare cu 150 de epoci si early stopping am obtinut 0.74 pe Kaggle

In continuare am incercat sa optimizam numarul de neuroni din cadrul layer-urilor dense, am incercat valori din multimea `[64, 128, 256, 512]` pentru cele doua layer-uri dense si am obtinut parametrii optimi de 512, respectiv 256. In continuare vom testa diverse valori pentru valorile dropout-ului. Initial vom incerca layere de Dropout doar dupa layerele de MaxPooling, din literatura fiind aratat ca acestea dau cele mai bune rezultate. Deci vom incerca modele de forma:

`CL32C7V~CL32C7V~BN~P2~DO[0-100] -> CL64C5V~CL64C5V~BN~P2~DO[0-100] -> F -> D128~DO[0-100] -> D128~DO[0-100] -> D3`. Am obtinut urmatoare valori pentru dropout: `{'Dropout': 0.36117159829314394, 'Dropout_1': 0.3640245915268092, 'Dropout_2': 0.22799073875606968, 'Dropout_3': 0.2741958309198733}` Cu o acuratete de `0.7604444622993469`.

Vom rula 100 de epoci cu aceste valori pentru a vedea evolutia acuratetii pe datele de validare. Pe baza acestui model am obtinut pe Kaggle o acuratete de `0.73538`.

```
100%|██████████| 100/100 [1:47:48<00:00, 64.69s/it, best loss: -0.7942222356796265]
{'Dropout': 0.0024711417287797896, 'Dropout_1': 0.1720349156837942, 'Dropout_2': 0.26467025953719037, 'Dropout_3': 0.37752757310387497, 'Dropout_4': 0.06978685281806113} <tensorflow.python.keras.engine.sequential.Sequential object at 0x7fc8b11b9d10>
```

Am incercat intervale de 5 si 10% in jurul valorilor de dropout => FAIL. Consideram ca dropout rate urile au fost optimizate.

In continuare vom incerca sa optimizam learning rate-ul.

Am gasit un optim de `0.0017`.

Din doc, epslion=1e-7 s-ar putea sa nu fie o valoare extrem de buna, un test cu valori de la 1e-8 la 1 arata ca o valoare de 0.001 ofera rezultate mai bune.

Cu modelul anterior si lr & epsilon optimizati am obtinut o acuratete de 0.76 pe Kaggle.

Am ditch-uit preprocesarea de mana si folosim ImageDataGenerator. La un prim test pare ca merge mai bine [44].



===============================================

Vom incerca optimizarea kernelurilor din layerele convolutionale. rula ciobinatii din multimea `{3, 5, 7, 9, 11}` pe modelul anterior pe 30 de epoci fiecare. (`BN -> CL32C{k1}V~CL32C{k1}V~BN~P2~DO18 -> CL64C{k2}V~CL64C{k2}V~BN~P2~DO82 -> F -> D512~DO26 -> D256~DO62 -> D3`). Ne uitam la top 5 modele care au suma `val_acc` cea mai mare [48], vom rerula aceste 5 modele pe 50 de epoci fiecare pentru a urmari evolutia [60-65]. Pare ca `(11, 3)` da cele mai bune rezultate. Incercam sa rulam cu lr si epsilon gasite anterior pentru a vedea cum merge.

Vom incerca reoptimizarea dropout-ului pe acest model.

Nu vom mai folosi keras, vom incerca toate valorile intre [0%, 100%] cu gap de 10% pentru fiecare dintre cele 4 layere de dropout.



================================================

Utilizand `Hyperas` am rulat o varietate de 50 de modele avand la baza arhitecutra VGG16 (2 layere convolutionale urmate de un layer de max pooling), fiecare pe cate 25 de epoci si am obtinut urmatorul model optimizat:

`CL64C5S~CL64C5S~BN~P2~DO50 -> CL128C11S~CL16C9S~BN~P2~DO50 -> CL24C7S~CL64C7S~BN~P2~DO50 -> F -> D256~DO50 -> D64~DO50 -> D3 `

Vom rula acest model p un numar mai mare de epoci pentru a urmari evolutia pe setul de validare.