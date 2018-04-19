#include "viewer.hpp"

int main(){
	Viewer v;

    v.init();

    while(true){
		if(!v.update()){
			std::cout << "Exit." << std::endl;
			break;
		}
	}

	return 0;
}
