data:
	g++ -std=c++17 -I/opt/homebrew/include -L/opt/homebrew/lib -o my_program main.cpp -lfftw3 -lm

plot:
	./my_program
	python3 plotgen.py

clean:
	rm *.dat
	rm -rf plots
	rm ./my_program

build: data plot clean
