## Mobile authentication of copy detection patterns: how critical is to know fakes?

The research was supported by the [SNF](http://www.snf.ch) project No.200021_182063. 

Keras implementation of ["Mobile authentication of copy detection patterns: how critical is to know fakes?"](http://sip.unige.ch/articles/2021/Taran_WIFS2021.pdf) 
##

Protection of physical objects against counterfeiting is an important task for the modern economies. In recent years, the high-quality counterfeits appear to be closer to originals thanks to the rapid advancement of digital technologies. To combat these counterfeits, an anti-counterfeiting technology based on hand-crafted randomness implemented in a form of copy detection patterns (CDP) is proposed enabling a link between the physical and digital worlds and being used in various brand protection applications. The modern mobile phone technologies make the verification process of CDP easier and available to the end customers. Besides a big interest and attractiveness, the CDP authentication based on the mobile phone imaging remains insufficiently studied. In this respect, in this paper we aim at investigating the CDP authentication under the real-life conditions with the codes printed on an industrial printer and enrolled via a modern mobile phone under the regular light conditions. The authentication aspects of the obtained CDP are investigated with respect to the four types of copy fakes. The impact of fakes’ type used for training of authentication classifier is studied in two scenarios: (i) supervised binary classification under various assumptions about the fakes and (ii) one-class classification under unknown fakes. The obtained results show that the modern machine-learning approaches and the technical capacity of modern mobile phones allow to make the CDP authentication under unknown fakes feasible with respect to the considered types of fakes and code design.
<p align="center">
<img src="http://sip.unige.ch/files/1716/3346/6384/indigo_mobile_main_diagram.png" height="250px" align="center">
<br/>
<br/>
Fig.1: General scheme of the CDP life cycle.
</p>

## [Dataset](http://sip.unige.ch/projects/snf-it-dis/datasets/indigo-mobile) 

- Industrial printer:
	- HP Indigo 5500 DS
	- resolution 812 dpi 	
- Mobile phone:
	- iPhone XS
	- automatic photo shooting settings
	- under the regular light conditions
- Hand-crafted fakes:
	- Fakes #1 white: RICOH MP C307 on the white paper
	- Fakes #1 gray: RICOH MP C307 on the gray paper
	- Fakes #2 white: Samsung CLX-6220FX on the white paper
	- Fakes #2 gray: Samsung CLX-6220FX on the gray paper

## Usage

#### 1. Install the dependencies
```bash
$ pip install -r requirements.txt
```
#### 2. Train the supervised step
```bash 
$ python ./supervised_classification/train.py
```
#### 3. Test the supervised step
```bash 
$ python ./supervised_classification/test.py
```
#### 4. Train the one-class classificaiton step
```bash 
$ python ./one_class_classification/train_....py
```
#### 5. Test the one-class classificaiton step
```bash 
$ python ./one_class_classification/test_....py && python ./one_class_classification/oc-svm_....py
```


## Citation
O. Taran, J. Tutt, T. Holotyak, R. Chaban, S. Bonev, and S. Voloshynovskiy, "Mobile authentication of copy detection patterns: how critical is to know fakes?," in Proc. IEEE International Workshop on Information Forensics and Security (WIFS), Montpellier, France, 2021. 

	@inproceedings { Taran2021IndigoMobile,
	    author = { Taran, Olga and Tutt, Joakim and Holotyak, Taras and Chaban, Roman and Bonev, Slavi and Voloshynovskiy, Slava },
	    booktitle = { IEEE International Workshop on Information Forensics and Security (WIFS) },
	    title = { Mobile authentication of copy detection patterns: how critical is to know fakes? },
	    address = { Montpellier, France },
	    pages = { },
	    month = { December },
	    year = { 2021 }
	}



