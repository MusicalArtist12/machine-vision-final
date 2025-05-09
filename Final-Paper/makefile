# Julia Abdel-Monem, 2025

default: main.pdf

edit:
	vim main.tex

main.pdf: main.tex
	pdflatex main
	bibtex main
	pdflatex main
	pdflatex main

clean: main.aux main.log main.pdf
	rm main.aux main.log main.pdf
