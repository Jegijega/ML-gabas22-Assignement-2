ML-gabas22-Assignement-2

კონკურსში ამოცანაა ვიწინასწარმეტყველოთ, არის თუ არა კონკრეტული ბანკის ტრანზაქცია თაღლითური

train: 590540 row, 434 column merge-ის შემდეგ
test: 506691 row, 433 column
fraud rate ძალიან დაბალი მხოლოდ 3.5%-ია

ჩემი მიდგომა:

1. ცალცალკე notebook ყველა მოდელისთვის: LogisticRegression, DecisionTree, RandomForest, GradientBoosting, XGBoost. თითოეულში ცალცალკე heading-ებად: Cleaning, Feature Engineering, Feature Selection, Training.
2. ცალცალკე MLflow ექსპერიმენტი ყველა არქიტექტურისათვის. ექსპერიმენტში ცალცალკე run-ები მაქვს Cleaning, FE, FS, Training-ისთვის.
3. მონაცემების სიდიდის გამო ყველა ექსპერიმენტი Kaggle- ზე ჩავატარე
4. საუკეთესო მოდელი (XGBoost) Pipeline-ად შევინახე და დავარეგისტრირე DagsHub-ის Model Registry-ში

რეპოზიტორიის სტრუქტურა:

├── model_experiment_LogisticRegression.ipynb
├── model_experiment_DecisionTree.ipynb
├── model_experiment_RandomForest.ipynb
├── model_experiment_GradientBoosting.ipynb
├── model_experiment_XGBoost.ipynb
├── model_inference.ipynb            
├── README.md
└── .gitignore

Feature Engineering:

კატეგორიული ცვლადების რიცხვითში გადაყვანა

კატეგორიული ცვლადები Label Encoding-ით გადავიყვანე. cat.codes-ით გადაყვანის შემდეგ NaN-ი "missing" კატეგორიად შევავსე. ჯერ One-Hot ვცადე, მაგრამ ფიჩერების რაოდენობა 1000+ გახდა და training შენელდა, ამიტომ Label Encoding ვარჩიე

FraudPrep-ში cat_maps dictionary ვინახავ. ტრენინგზე ისწავლის mapping-ს, test-ზე იმავეს იყენებს. ახალი მნიშვნელობები test-ში -1-ით ჩანაცვლდება.

NaN მნიშვნელობების დამუშავება

NaN ბევრი იყო ამიტომ მივმართე შემდეგ მიდგომას:
numeric-ისთვის LogisticRegression-ისთვის მედიანა ვცადე, ხეებისთვის -999.
categorical-ისთვის "missing" ცალკე კატეგორიად. ეს უფრო ლოგიკურად მივიჩნიე რადგან NaN ხშირად ნიშნავდა რომ ეს ფიჩერი არ ჰქონდა და ვფიქრობ ასე გადაწყვეტა ყველასზე სწორი იყო.

Cleaning მიდგომები:

ყველაზე დიდი პრობლემა იყო ცარიელი სვეტები. გავატესტე 3 threshold სხვადასხვა notebook-ში:

თავდაპირველად threshold ავიღე 0.7 რომელმაც დადროპა 208 სვეტი და დამრჩა 226. LR ვერ მუშაობს NaN-თან, ბევრი imputed -999 noise-ი ხდება ამიტომაც ავარჩიე რომ ეს threshold დამეტოვებინა, თუმცა სხვებში 0.8 და 0.9 ვცადე თუმცა საბოლოოდ 0.9 გამოვიყენე. ხეები კარგად მუშაობენ sparsity-თან და ამიტომ ავარჩიე 0.9.

Feature Engineering ნაბიჯები:

TransactionAmt_log, heavily right-skewed იყო (mean=$135, max=$31937), log-მა უფრო ნორმალური ფორმა მისცა. თუმცა ეს უფრო LR-ისთვისაა გაკეთებული. ხეებისთვის log redundant-ია, ამიტომ დანარჩენ notebook-ებში ამოვიღე.

ემაილის domain-ის გასუფთავება ვცადე (gmail.com → gmail), მაგრამ შედეგი არ გააუმჯობესა P_emaildomain და R_emaildomain ისედაც ხშირად ცარიელი იყო. ამიტომ დავანებე თავი.

Feature Selection:

1. Correlation Drop (correlation > 0.95)
numeric სვეტებს შორის correlation დავთვალე და სადაც corr > 0.95 იყო, წყვილიდან ერთს ვაგდებდი. LR-ისთვის განსაკუთრებით საჭიროა, რადგან multicollinearity coefficient-ებს არასტაბილურს ხდის.

2. Tree-based Feature Importance
ეს მიდგომა მხოლოდ ხეებისთვის გავტესტე
პატარა "scout" RandomForest (n_estimators=50, max_depth=10) ვწვრთნი 100k სემპლზე
feature_importances-თი ვიღებ ყველა feature-ის მნიშვნელობას
top 100 ფიჩერს ვტოვებ.
ეს უკეთესია ვიდრე correlation drop, რადგან correlation მხოლოდ "მსგავსებას" ხედავს, არ იცის ფიჩერი useful-ია თუ არა target-ისთვის. Importance პირდაპირ გვეუბნება, რომელი ფიჩერია ყველაზე გამოსადეგი fraud detection-ისთვის.

Training:

ტესტირებული მოდელები:
LogisticRegression
DecisionTree
RandomForest
GradientBoosting
XGBoost

ცალცალკე ცხრილებში ყველა მოდელის შედეგები:

LogisticRegression

LogisticRegression-ში 3 სხვადასხვა C გავტესტე:
C=0.01-ით AUC გამოვიდა 0.8269 (std 0.0013) - ძალიან ძლიერი regularization, underfit-ია.
C=1.0-ით AUC გამოვიდა 0.8282 (std 0.0007) - საუკეთესო შედეგი მომცა.
C=100-ით AUC გამოვიდა 0.8261 (std 0.0006) - სუსტი regularization, overfit.

DecisionTree

აქ 2 hyperparameter ვცადე max_depth და class_weight:
max_depth=3-ში მოდელი ძალიან ცოტას სწავლობდა (underfit)
max_depth=10-ში გაცილებით უკეთესი შედეგი მივიღე
max_depth=None-ში (ხე იზრდება ბოლომდე) მოდელი overfit-ი ხდება და AUC ეცემა (0.7288).
ანუ საუკეთესო კომბინაცია გამოვიდა max_depth=10 + class_weight="balanced" AUC-ით 0.8321.

RandomForest

აქაც იგივე 2 hyperparameter გამოვიყენე max_depth და class_weight
max_depth=5-ში default-ით AUC 0.8340, balanced-ით 0.8459.
max_depth=15-ში default-ით 0.8748, balanced-ით 0.8828.
max_depth=None-ში default-ით 0.8887, balanced-ით 0.8960(საუკეთესო).

GradientBoosting

აქ 2 hyperparameter გამოვიყენე learning_rate და max_depth.
lr=0.05-ით d=3-ზე 0.8416, d=7-ზე 0.8576. ცოტა underfit.
lr=0.1-ით d=3-ზე 0.8556, d=7-ზე 0.8723. საუკეთესო.

XGBoost
XGBoost-შიც learning_rate და max_depth გამოვიყენე
lr=0.1-ით d=3-ზე 0.8794, d=6-ზე 0.9002, d=10-ზე 0.9145.
lr=0.3-ით d=3-ზე 0.8885, d=6-ზე 0.9028, d=10-ზე 0.9065.

საუკეთესო გამოვიდა lr=0.1 + max_depth=10 AUC-ით 0.9145. ეს იყო საუკეთესო შედეგი ყველა მოდელს შორის. რეგულარიზაცია XGBoost-ში ძალიან დამეხმარა რომ არ მომხდარიყო overfit.

XGBoost გამოვიდა საუკეთესო 3 მიზეზით
1. ყველაზე მაღალი CV AUC (0.9145)
2. multi-threaded სწრაფი
3. regularization აკონტროლებს overfit-ს

MLflow Tracking URI: https://dagshub.com/gabas22/ML-gabas22-Assignement-2.mlflow
ექსპერიმენტების ბმული: https://dagshub.com/gabas22/ML-gabas22-Assignement-2/experiments
Model Registry: https://dagshub.com/gabas22/ML-gabas22-Assignement-2/models , ieee-fraud-best, version 1

Kaggle leaderboard score (xgboost submission):
Public Score: 0.9258
Private Score: 0.8889
