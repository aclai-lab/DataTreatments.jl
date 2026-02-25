# Datatreatment

Datatreatment è una struttura che ospita
- dataset (supervisionati e non)
- metadati

I possibili dataset individuati sono:
- tabulari
- aggregazione (dataset multidimensionali tabularizzati)
- riduzione (dataset multidimensionali di dimensioni ridotte)

La struttura è unica, quello che cambia è la struttura del metadato.

Casi speciali:
- dataset con valori mancanti
- dataset con colonne di tipo misto

### terminologia
variabili = nome delle colonne, dette anche features

## dataset tabulari
Caso base
- il dataset è caricato come matrice
- i metadati contengono
    - i nomi delle variabili
    - il tipo di variabile

## dataset aggregati
I dataset devono essere tabularizzati con la seguente procedura:
- le istanze multidimensionali vengono finestrate
- ad ogni finestra viene applicata una, o una serie di funzioni di riduzione,
  tipicamente maximum e mean
- i metadati dovranno contenere:
    - nomi variabili
    - tipo variabile
    - numero finestra associata
    - funzione di riduzione utilizzata

## dataset ridotti
I dataset vengono finestrati per ridurne la dimensione
Da notare che il dataset non cambia di dimensionalità, a differenza del caso aggregato.
- i metadati potrebbero contenere unicamente il nome delle variabili

* da notare che potrebbe essere utile salvare anche le funzioni di riduzione che
  verranno applicate in un secondo momento, secondo la logica di funzionamento di
  ModaldecisionTrees

## groupby
Necessario per partizionare il dataset in gruppi omogenei, per futuri calcoli o funzioni.

* dobbiamo suddividere tra tipi discreti e tipi continui,
  potrebbe tornare utile CategoricalArray, vedi SoleXplorer

* potrebbe essere interessante estendere la struttura del DataTreatment:
- dataset
- metadati vettore colonna
- metadati generali (che potrebbero includere indici di groupby)

### esempi di groupby



