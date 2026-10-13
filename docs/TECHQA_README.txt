The TechQA Dataset
##################

License
=======
The TechQA Dataset is released under the Community Data License Agreement - Permissive - Version 1.0.
Please see the CDLA-Permissive-v1.0.pdf file in the top-level directory


training_and_dev: Training and Development Data
=============================================

The training_and_dev directory contains the following files:
 - training_Q_A.json           the training set
 - dev_Q_A.json                the development set
 - training_dev_technotes.json documents referenced in the training and development set


training_Q_A.json and dev_Q_A.json contain lists of maps.  Each maps represents a labeled instance.
----------------------------------
  - training_Q_A.json contains 600 labeled instances
  - dev_Q_A.json contains 310 labeled instances.

Each instance consists of the following fields:

These fields are system inputs
QUESTION_ID:     the unique id of the instance
QUESTION_TITLE:  the title of the question - a summary, introduction, and occasionally a full question
QUESTION_BODY:   typically a longer span of text than the QUESTION_TITLE, contains the actual question asked by the user. Occasionally it is identical to the text
DOC_IDS:         a list of strings: each is a unique document id, found in training_dev_technotes.json
		 systems are asked to find an answer to the question among these documents, if available

These fields are system answers
ANSWERABLE:      'Y' if human annotators found an answer in one of the documents in DOC_IDS
		 'N' otherwise
DOCUMENT:	 if ANSWERABLE is 'Y': the id of the document containing the answer
                 if ANSWERABLE is 'N': the value is '-'
START_OFFSET:    if ANSWERABLE is 'Y': zero-based character offset of the first character of the answer in the 'text' field of the document having 'id' == DOCUMENT
                 if ANSWERABLE is 'N': the value is '-'
END_OFFSET:      if ANSWERABLE is 'Y': zero-based character offset of the end of the answer in the 'text' field of the document having 'id' == DOCUMENT, using C/PYTHON conventions
                 if ANSWERABLE is 'N': the value is '-'
ANSWER: 	 if ANSWERABLE is 'Y': the string document['text'][START_OFFSET:END_OFFSET] where document is the document having 'id'=DOCUMENT
		 if ANSWERABLE is 'N': the value is '-'

training_dev_technotes.json
---------------------------
See 'Technote Collection' below for a description of technotes.  This file contains a subset of the collection.
Structurally, the field contains a map from document id to documents.  
Each document contains the following fields
 'id':            unique document id
 'content':       html version of the page
 'title':         title of the page, detagged
 'text':          automatically detagged body of the page
 'metadata':      map with the following fields:
                  'sourceDOcumentId':   same as 'id' in most cases
		  'date':               date of last modification
		  'productName':        name of the product for which the technote document was written
		  'productId':          id of the product, could be useful for cross-referencing or disambiguating products
		  'canonicalUrl':       canonical URL of the page - does not necessarily correspond to the actual URL of the page
		  

validation: verifying that a Q/A system is running
====================================================
The validation folder contains files that will be used to verify that a submitted system is running correctly.
Specifically, when a submission is received, the system will be run against the validation_questions, and the output scored against validation_reference.json.
The scores produced by the evaluation.py script will be returned to the submitter who can check them against a local run.
If the submitter decides that the scores are correct, then the submitter can trigger the official evaluation

The validation folder contains the following files:
 - validation_questions.json         input file for the system, containing the questions arranged as an array of maps
                       		       Each maps contains the QUESTION_ID, QUESTION_TITLE, QUESTION_TEXT, DOC_IDS fields described above.
 - validation_reference.json         ground truth against which the system output is evaluatated. The file has the same format as the training/dev sets described above
 - validation_technotes.json         The collection of technotes mentioned in validation_questions.json. Has the same format as training_dev_technotes.json
 - validation_example_output.json    Example output of a system.  The scorer expects the system output to be in this format. See 'Output Format' below in the 'Leaderboard' section.
                                       This specific file is not the actual output of a system but was randomly generated using the ground truth.

technote_corpus
===============
The technote corpus contains the collection of the IBM technotes available on the web as of April 4, 2019 - 801998 technotes in total.
Technotes are IBM documents maintained by IBM support personnel that contain information about common questions asked by customers, including  product  upgrade  information and solutions to problems.
They cover a broad variety of products, including many that are not produced by IBM.  

We believe this to be a useful resource, for example for fine-tuning language models for the technical support domain.

data format
-----------
The file full_technote_collection.txt.bz2 contains one json array per line. Each entry of the array is a map with the fields described above in the training_dev_technotes.json section.
The file can be read using the reader.py example code, which produces a dictionary indexed by the document 'id'

technote_corpus license
-----------------------
This corpus is released separately from the data in the training_and_dev and validation dataset, also under the Community Data License Agreement - Permissive - Version 1.0 license.
A copy of the license is in the folder.

