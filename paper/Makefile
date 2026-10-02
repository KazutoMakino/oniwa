.PHONY: all clean arxiv

PDF = main.pdf
SRC = main.tex main.bib

all: $(PDF)

$(PDF): $(SRC)
	pdflatex -interaction=nonstopmode main.tex
	bibtex main
	pdflatex -interaction=nonstopmode main.tex
	pdflatex -interaction=nonstopmode main.tex

arxiv: $(PDF)
	@echo "Packaging for arXiv submission..."
	tar -czvf oniwa-paper-arxiv.tar.gz main.tex main.bbl $(shell find figures -type f 2>/dev/null)
	@echo "arXiv package created: oniwa-paper-arxiv.tar.gz"

clean:
	rm -f *.aux *.bbl *.blg *.fdb_latexmk *.fls *.log *.out *.synctex.gz *.toc page-*.png page-test-*.png
