Data Science Bowl 2017

Problem Discussion
In this paper, we attempt to layout the strategies to compete in the Data Science Bowl 2017 contest on Kaggle. The stated goal of competition is to develop a machine learning model which can predict the presence of lung cancer from a set of lung images. The lung images are in the ‘dicom’ format – a standard widely used for medical image processing, storage and analysis. The image data consists of train and test set. The data for each patient consists of multiple ‘dicom’ files, all of which represent a single slice (cross-section) of lungs. The files can be combined to create a three-dimensional image of lungs of each patient. We discuss below our approach to resize the data to a more manageable size and methods to do exploratory data analysis, data visualization and feature selection before applying predictive techniques.

Significance
In the United States, lung cancer strikes 225,000 people every year, and accounts for $12 billion in health care costs. Early detection is critical to give patients the best chance at recovery and survival. A predictive model having good degree of accuracy can be scaled in application to thousands of patients in areas where medical facilities are not well established. From the perspective of data scientists analyzing hundreds of thousands dicom images proves to be a worthwhile challenge which helps in developing image recognition techniques.

Literature Overview
We now discuss different research papers, journals, conferences and publications where various machine learning models were used for lung cancer detection.
1. Lung Cancer Cell Identification based on artificial neural network ensembles, Artificial Intelligence in Medicine (2002); Zhou Et al.
In this paper, a procedure named Neural Ensemble based detection is used which utilizes an artificial neural network ensemble to identify lung cancer cells in the images of biopsies of the subjects to be diagnosed. The ensemble is built on two level ensemble architecture. The first level ensemble is used to judge whether the cell is normal with high confidence where each individual network has only two outputs respectively normal cell or cancer cell. The predictions of those individual methods are combined by a novel method of full voting which judges a cell to be normal only when all individual networks judge it to be normal. The second level ensemble is used to deal with cells that are judged to be cancerous by first level ensemble where each individual network has five outputs among which four are type of cancer cells and one represents normal cell. The predictions of those individual networks are combined by prevailing method i.e. plurality voting. The researchers were able to achieve not only high rate of overall identification but also a low rate of false negative identification.
2. Diagnosis of Lung Cancer Prediction System Using Data Mining Classification Techniques; V. Krishnaiah et al, / (IJCSIT) International Journal of Computer Science and Information Technologies, Vol. 4 (1), 2013, 39 - 45
In this study, authors examine the potential use of classification based data mining techniques such as Rule based, Decision tree, Naïve Bayes and Artificial Neural Network to massive volume of healthcare data. For data preprocessing and effective decision making One Dependency Augmented Naïve Bayes classifier (ODANB) and naive creedal classifier 2 (NCC2) were used. This is an extension of naïve Bayes to imprecise probabilities that aims at delivering robust classifications also when dealing with small or incomplete data sets. The system extracts hidden knowledge from a historical lung cancer disease database. The most effective model to predict patients with Lung cancer disease appears to be Naïve Bayes followed by IF-THEN rule, Decision Trees and Neural Network. Decision Trees results are easier to read and interpret. The drill through feature to access detailed patients’ profiles is only available in Decision Trees. Naïve Bayes fared better than Decision Trees as it could identify all the significant medical predictors. The relationship between attributes produced by Neural Network is more difficult to understand. Lung cancer prediction system can be further enhanced and expanded. It can also incorporate other data mining techniques, e.g., Time Series, Clustering and Association Rules.
3. Applications of Machine Learning in Cancer Prediction and Prognosis, Departments of Biological Science and Computing Science, Joseph A. Cruz, David S. Wishart, University of Alberta Edmonton, AB, Canada T6G 2E8
In assembling this review, the authors conducted a broad survey of the different types of machine learning methods being used, the types of data being integrated and the performance of these methods in cancer prediction and prognosis. A number of trends were noted, including a growing dependence on protein biomarkers and microarray data, a strong bias towards applications in prostate and breast cancer, and a heavy reliance on “older” technologies such artificial neural networks (ANNs) instead of more recently developed or more easily interpretable machine learning methods. 
4. Automatic segmentation of lung nodules with growing neural gas and support vector, Computers in Biology and Medicine, Stelmo Magalhaes Barros Netto, Aristofanes Correa Silva, Rodolfo Acatauassu Nunes, Marcelo Gattass
In this paper, the proposed method consists of the acquisition of computerized tomography images of the lung, the reduction of the volume of interest through techniques for the extraction of the thorax, extraction of the lung, and reconstruction of the original shape of the parenchyma. After that, growing neural gas (GNG) is applied to constrain even more the structures that are denser than the pulmonary parenchyma (nodules, blood vessels, bronchi, etc.). The next stage is the separation of the structures resembling lung nodules from other structures, such as vessels and bronchi. Finally, the structures are classified as either nodule or non-nodule, through shape and texture measurements together with support vector machine.
5. Lung Nodule Detection using a Neural Classifier, M. Gomathi and Dr. P. Thangaraj, IACSIT International Journal of Engineering and Technology, Vol.2, No.3, June 2010
This paper discusses a dot-enhancement filter for nodule candidate selection and a neural classifier for false-positive finding reduction. The performance is evaluated as a fully automated computerized method for the detection of lung nodules in screening CT in the identification of lung cancers that may be missed during visual interpretation.

Data Mining and Data Cleaning
In this dataset, we are given over a thousand low-dose CT images from high-risk patients in DICOM format. Each image contains a series with multiple axial slices of the chest cavity. Each image has a variable number of 2D slices, which can vary based on the machine taking the scan and patient. The DICOM files have a header that contains the necessary information about the patient id, as well as scan parameters such as the slice thickness. The images in this dataset come from many sources and vary in quality. Older scans were imaged with less sophisticated equipment.
As with any other problem dealing with data science, we start by performing an Exploratory Data Analysis (EDA). EDA is an approach/philosophy for data analysis that employs a variety of techniques (mostly graphical) to
a. Maximize insight into a data set;
b. Uncover underlying structure;
c. Extract important variables;
d. Detect outliers and anomalies;
e. Test underlying assumptions;
f. Develop parsimonious models; and
g. Determine optimal factor settings.1
Since the data was in DICOM image format, read the data using ORODICOM libraries and host of other medical imagining libraries from the BioCGenerics website. The DICOM image consists of two parts – image and a header. The image is 512x512 pixels in resolution and the header contains meta data of that image in a 42x7 matrix. We resize the image size to 32x32 pixels to better handle large number of images without affecting performance of the machine. We then visualize the images using the ‘image’ function in above library.
We intend to have all the data for a person in a single row. Thus, we create a 2D matrix by stacking additional row for each person. There are variable number of DICOM images per person and so we decide to choose only the images having highest variability. To achieve this, we create 3 new images for each person by performing PCA and choosing the top 3 images(components) having highest variability. These are also known as Eigen vectors. Eigen vectors are widely used in face recognition and for simplicity here we will refer to our PCA components as Eigen Lungs. We then serialize or flatten each Eigen lung by creating a 1D Vector having the data of all Eigen Lungs for a single person. Our resulting final 2D matrix can thus hold entire data of all patients where each row represents a patient and each column represents a pixel value.

Data Visualization
The DICOM file itself is a single image of 512x512 pixels. We can image it using the orodicom library image function. Each image is actually a cross sectional slice of the lungs. Given below is a sample image retrieved from the orodicom file.
Thereafter we also performed PCA and created new eigen vector image having maximum variance from the first three eigen vectors. The resulting Eigen Lungs looked nothing like the original images, however they contained important data that can be fed to a predictive model.

Application of Predictive Techniques
We tried 3 different models on our data object (2D matrix), namely Random Forest, Neural Networks and GLM. We now discuss each of these models, and how we used them.

Predictive Technique #1 Random Forest
A random forest is a multi-way classifier which consists of number of trees, with each tree grown using some form of randomization [2]. The leaf nodes of each decision tree are labelled by estimates of the posterior distribution over the image classes [2].
Each internal node contains a test that best splits the space of data to be classified [2]. An image is classified by sending it down every tree and aggregating the reached leaf distributions [2]. Randomness can be injected at two points during training: in subsampling the training data so that each tree is grown using a different subset; and in selecting the node tests - Formulation / Libraries [2].
As per the given problem statement, we are required to not only classify the given test samples but also predict the probabilities of each sample belonging to each species. This probability gives some kind of confidence on the prediction. Given below are details of classifier and calibrator hyperparameter values we tuned after trial and error to achieve optimal model performance:
Libraries(R):
We used randomForest function from randomForest library to model and predict probabilities of acquiring cancer by each patient.

Predictive Technique #2 GLM
The general linear model is a statistical linear model
Given mathematically as
Y = XB+U
where Y is a matrix with series of multivariate measurements, X is a matrix that might be a design matrix, B is a matrix containing parameters that are usually to be estimated and U is a matrix containing errors or noise.
Thus, GLM is an ANOVA procedure in which the calculations are performed using a least squares regression approach to describe the statistical relationship between one or more predictors and a continuous response variable. Predictors can be factors and covariates. GLM codes factor levels as indicator variables using a 1, 0, - 1 coding scheme, although you can choose to change this to a binary coding scheme (0, 1). Factors may be crossed or nested, fixed or random. Covariates may be crossed with each other or with factors, or nested within factors. The design may be balanced or unbalanced. GLM can perform multiple comparisons between factor level means to find significant differences.
Given below are details of GLM model we tuned after trial and error; and our optimal model performance for GLM:
Libraries(R):
1. ‘glm’ function was used with family=binomial(link='logit').
2. The model did not converge and gave warning as such.

Predictive Technique #3 Neural Network
Artificial neural networks were designed to be modelled after the structure of the brain. They were first devised in 1943 by researchers Warren McCulloch and Walter Pitts [3]. Backpropagation, discovered by Paul Werbos [4], is a way of training artificial neural networks by attempting to minimize the errors. This algorithm allowed scientists to train artificial networks much more quickly. Each artificial neural network consists of many hidden layers. Each hidden layer in the artificial neural network consists of multiple nodes. Each node is linked to other nodes using incoming and outgoing connections. Each of the connections can have a different, adjustable weight. Data is passed through these many hidden layers and the output is eventually interpreted as different results.
There are three input nodes. Each input node represents a parameter from the dataset being used. Ideally the data from the dataset would be pre-processed and normalized before being put into the input nodes.
There is only one hidden layer in this example and it is represented by the nodes in blue. This hidden layer has four nodes in it. Some artificial neural networks have more than one hidden layer.
The output layer in this example diagram is shown in green and has two nodes. The connections between all the nodes (represented by black arrows in this diagram) are weighted differently during the training process. We are using the Keras package from python to set up the neural network. For Windows, Keras runs on top of Theano library. In first model, we used an input layer, an output layer and one hidden layer. There are some dropouts in between each layer. Since the output is multinomial, the output layer activation of ‘softmax’ type. In another model, we used only an input and an output layer.

We uploaded the predicted probabilities to Kaggle under the team name – ‘TTU_Max_Pradnya’. Our Kaggle log loss scores for the three models are tabulated below:

Model                 Log Loss Private     Log Loss Public
Random Forest            0.62557                0.70982
General Linear Model    16.78611               23.02612
Neural Net               0.72469                0.48856

Learnings and Limitations
We now discuss our learnings in terms of advantages and limitations of each model.
Overall Competition
Learnings
• We learned to work out and reduce huge datasets.
• How to use HPCC for queuing R jobs on Hrothgar cluster.
• We need worker nodes assigned to us on the Janus cluster to run R code efficiently.
Limitations
• Huge dataset takes up most of the memory allocated to us on the clusters.
• Need to work on parts of dataset at a time.
• If the nodes are not available, Hrothgar queue may take up to 2 days to run the R job.

Random Forest
Features
• It’s one of the most accurate learning algorithms available
• Hyper parameter tuning relatively easy
Limitations
• Can attempt to fit a really complex tree to the data, leading to overfitting
• Intuitively easy to understand, but difficult to get an insight as to what the algorithm does.

General Linear Model
Features
• The multivariate tests of significance can be employed when responses on multiple dependent variables are correlated.
• It can analyze effects of repeated measure factors.
Limitations
• It’s hypothesis space is limited and we can't solve non-linear problems since its decision surface is linear.

Neural Network
Features
• Few assumptions need to be verified for constructing models
• ANNs can model complex nonlinear relationships between independent and dependent variables, and so they allow the inclusion of many variables
Limitations
• Greater computational burden, proneness to overfitting, and the empirical nature of model development
• Tuning the hyper parameters is challenging. Tuning neural networks need high computing power. The key issue lies in the selection of proper image input features to attain high efficiency with less computational complexity

Conclusions
We observed better performance for neural network using H2O package in R. Our best performing neural network gave us a log loss of 0.48856 on Kaggle along with a rank of 32 on Public leaderboard (323 on Private leaderboard). Random Forest gave good accuracy and ran efficiently over our data set and outperformed Neural Net in private leaderboard. As future work, we plan to use feature extraction algorithms to clean dataset and retain important information thereby avoid overfitting. We also plan to study other boosting techniques like Adaptive Boosting, Extreme Gradient Boosting along with different types of ensembles.
 
