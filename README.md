# RADio-

The original RADio framework proposed rank-based divergence metrics to measure normative diversity in news recommendations. The accompanying paper showcased the behavior of these metrics on the Microsoft News Dataset and its accompanying news recommenders. To do justice to the normative underpinnings of the metrics a great deal of complicated data preprocessing was necessary, which made the code slow and hard to interpret, and therefore hard to reuse in a different application. The repository released here comprises a refactoring of the RADio code. It is stripped down to its essential components and accompanied with clear documentation, which should allow for easier understanding and adoption. 

## Usage
Install the packages listed in the requirements.txt in a Python 3.9 environment. Follow the steps in the Notebook (diversity.ipynb) for example usage and configuration of the metrics. **Note that these metrics should serve as examples, and are likely not plug-and-play solutions for your application.**  

## Data
Ensure that the data folder contains the following files:
- news.tsv <br>
    Tab-separated file as supplied in the MIND dataset. To be usable, entities need to have a Label and a Type. Example: <br>
    N12733	news	politics	"Here comes the title"  "Here comes the subtitle"		url	[{"Label": "Entity Label from title", "Type": "O"]	[{"Label": "Entity Label from subtitle", "Type": "O"] <br>
- recommendations.json  <br>
    JSON files containing the generated recommendations that reads into a DataFrame. Should contain the fields impr_index, userid, date and history (optionally ordered by recency). The other columns will be interpreted as 'recommendation' columns, and their names as the name of the algorithm used. Example: <br>
     impr_index  userid  date        history         lstur           random <br>
     34          U1234   13-08-2023  [N5, N4, N3],   [N1, N2, N3],   [N3, N2, N1] <br>
<br>    
The news.tsv follows the same format as the MIND dataset. The recommendations can be constructed from training the [Microsoft Recommenders](https://github.com/microsoft/recommenders), and merging those with the relevant information (userid, date, history) from the MIND behavior file. For more details about the MIND format, see [the MIND website](https://github.com/msnews/msnews.github.io/blob/master/assets/doc/introduction.md). The repository currently contains a sample of 100 rows and predictions, which correspond to MINDsmall_dev. Reach out to us for an example of the full file.

## Other relevant work
Refer to the following papers for more background information on the metrics and their normative underpinnings:

Sanne Vrijenhoek, Gabriel Bénédict, Mateo Gutierrez Granada, and Daan Odijk. 2023. RADio* – An Introduction to Measuring Normative Diversity in News Recommendations. ACM Trans. Recomm. Syst. (December 2023). https://doi.org/10.1145/3636465

Sanne Vrijenhoek, Mesut Kaya, Nadia Metoui, Judith Möller, Daan Odijk, and Natali Helberger. 2021. Recommenders with a Mission: Assessing Diversity in News Recommendations. In Proceedings of the 2021 Conference on Human Information Interaction and Retrieval (CHIIR '21). Association for Computing Machinery, New York, NY, USA, 173–183. https://doi.org/10.1145/3406522.3446019

## Acknowledgments
This repository builds upon the code of the original RADio framework, which was a collaboration between Sanne Vrijenhoek, Gabriel Bénédict and Mateo Gutierrez Granada. 


