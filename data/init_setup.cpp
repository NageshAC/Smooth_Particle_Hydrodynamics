#include <fstream>
#include <iostream>
#include <iomanip>

#define X_Step 8
#define Y_Step 8
#define Z_Step 8

#define X_Step_length 1
#define Y_Step_length 1
#define Z_Step_length 1


#define Mass 0.5
#define Scope_Length 0.5
#define Start_Position {0., 0., 0}

void build_square(const std::string & );

int main(int argc, char *argv[]){
    if (argc == 1) {
        // std::cout << "USAGE: \n\"" << "<exec_file> <out_file_path>" << "\"";
        build_square("data/init_setup.txt");
        return 0;
    }

    build_square(argv[1]);
    return 0;
}

void build_square(const std::string & file_path){
    // std::string file_path = "data/init_setup.txt";

    std::ofstream file(file_path);

    if(!file) { 
        std::cout << "The file: " << file_path << " cannot be opened" << std::endl;
        exit(-1);
    }

    std::cout << "Writing file " << file_path << "......." << std::endl;

    file << "## setup file for SPH \n\n\n";

    file << "mass                0.05\n";
    // file << "density             0.005\n";
    file << "scope_length        0.05\n";

    file << std::endl << std::endl;

    // Print positions and velocities
    file << "position " << X_Step * Y_Step * Z_Step << std::endl;
    for (int i = 0; i < X_Step; i++)
        for (int j = 0; j < Y_Step; j++)
            for (int k = 0; k < Z_Step; k++)
                file << std::setprecision(6) << std::fixed 
                     << X_Step_length * i << "    " << Y_Step_length * j << "    " << Z_Step_length * k 
                    //  << "    0.    0.    0. \n";
                     << std::endl;
    file.close();

    std::cout << "Done !" << std::endl;
}