

**Using Transformers**

**in Image**

ZainabꢀMohammad

RajitꢀPuzhakkarezhath

AviralꢀMehrotra

**Classification**

NicholasꢀChiu





**Introduction**

◉ Transformersꢀwereꢀintroducedꢀinꢀ2017ꢀandꢀhaveꢀbecomeꢀ

widelyꢀusedꢀinꢀNLPꢀapplications

◉ Inꢀ2021,ꢀvisionꢀtransformersꢀwereꢀadaptedꢀforꢀcomputerꢀ

visionꢀapplications

“ANꢀIMAGEꢀISꢀWORTHꢀ16X16ꢀWORDS:ꢀTRANSFORMERSꢀFORꢀIMAGEꢀRECOGNITIONꢀATꢀSCALE”ꢀ [http://arxiv-export-](http://arxiv-export-lb.library.cornell.edu/pdf/2010.11929)

[lb.library.cornell.edu/pdf/2010.11929](http://arxiv-export-lb.library.cornell.edu/pdf/2010.11929)

◉ Visionꢀtransformersꢀ(ViT)ꢀareꢀscalableꢀandꢀoutperformꢀwhileꢀ

stillꢀbeingꢀrelativelyꢀcheapꢀtoꢀfine-tune





**Problem**

◉ Motivation:ꢀImageꢀclassificationꢀnowꢀtypicallyꢀusesꢀaꢀCNN,ꢀ

soꢀwhatꢀisꢀtheꢀbestꢀmovingꢀforward?

◉ **Goal:**ꢀBuildꢀandꢀtrainꢀourꢀownꢀvisionꢀtransformerꢀmodelꢀandꢀ

compareꢀitꢀtoꢀCNNꢀmodelsꢀ(ourꢀownꢀandꢀResNet50)

◉ Willꢀourꢀresultsꢀshowꢀsimilaritiesꢀwithꢀtheꢀoriginalꢀpaper?ꢀ

Whatꢀdoesꢀthisꢀsayꢀaboutꢀtheꢀperformanceꢀandꢀapproachꢀofꢀ

usingꢀtransformersꢀinꢀtheꢀdomainꢀofꢀcomputerꢀvision?





**Dataset**

◉ EuroSAT

◉ 27,000ꢀlabelledꢀ64x64ꢀsatelliteꢀimages

◉ Classifiedꢀintoꢀ10ꢀdifferentꢀlandꢀuseꢀclassesꢀ(e.g.ꢀhighway,ꢀforest,ꢀ

river,ꢀetc)

◉ Usingꢀrealꢀmapꢀdataꢀasꢀopposedꢀtoꢀsimplerꢀimagesꢀbetterꢀ

reflectsꢀtheꢀviabilityꢀofꢀusingꢀtransformersꢀinꢀimageꢀ

classification.

AnnualꢀCrop

Forest

HerbaceousꢀVegetation





**Approach**

● First,ꢀweꢀconvertꢀtheꢀimagesꢀtoꢀpixelꢀ

values.ꢀ

● Weꢀthenꢀperformꢀdataꢀaugmentationꢀ

andꢀweꢀtakeꢀimageꢀpatchesꢀandꢀ

convertꢀthemꢀtoꢀ1Dꢀsequences.ꢀ

● Positionalꢀembeddingsꢀareꢀaddedꢀtoꢀ

theseꢀsequences.





**Vision Transformer**





**CNN Model**

◉ 5-layerꢀCNNꢀasꢀbaselineꢀtoꢀ

compare

◉ 3x3ꢀkernel

◉ ReLUꢀactivation

◉ MaxPooling

◉ Dropoutꢀofꢀ10%ꢀtoꢀpreventꢀ

overfitting





**CNN Results**





**Vision Transformer Results**





**Evaluation**

Our CNN (25 epochs) ResNet50 (25 epochs) Vision Transformer (3

epochs)

Trainable Parameters

Time per Epoch

157,018

8-9s

17,132,490

22-23s

85,806,346

\>=5000s (83 mins)

98.46%

Validation Accuracy

Validation Loss

87.20%

0.4658

94.69%

0.2988

0.0728





**Lessons Learned**

◉ Theꢀfundamentalꢀapproachꢀwasꢀtoꢀcreateꢀanꢀimageꢀclassificationꢀ

problemꢀbyꢀusingꢀimageꢀpatchesꢀasꢀtokensꢀinꢀaꢀsequence,ꢀandꢀthenꢀ

processingꢀitꢀwithꢀaꢀtransformer.

◉ Needꢀtoꢀbeꢀtrainedꢀonꢀlargeꢀamountsꢀofꢀdata

◉ ViTꢀrequiresꢀhugeꢀamountsꢀofꢀcomputationalꢀresourcesꢀandꢀtimeꢀ

whenꢀtrainingꢀfromꢀscratch

◉ Hasꢀtheꢀpotentialꢀtoꢀbeatꢀotherꢀstate-of-the-artꢀCNNꢀ

architecturesꢀwithꢀaꢀmoreꢀpowerfulꢀmachineꢀandꢀpretrainedꢀ

data





**Future Work**

◉ Currentꢀtrendsꢀareꢀsuggestingꢀaꢀshiftꢀtoꢀtheꢀtransformerꢀarchitecture.

◉ AꢀstudyꢀinꢀJuneꢀ2021ꢀaddedꢀaꢀtransformerꢀbackendꢀtoꢀResNetꢀ

whichꢀreducedꢀcostsꢀandꢀincreasedꢀaccuracy.ꢀ

◉ Teslaꢀisꢀnowꢀalsoꢀusingꢀtransformersꢀinꢀitsꢀautopilotꢀsystem.

◉ Futureꢀcomputerꢀvisionꢀcouldꢀmoveꢀtowardsꢀtransformersꢀinsteadꢀofꢀ

CNNs





**Thank You!**

**Questions?**

