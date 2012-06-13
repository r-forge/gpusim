
#include <iostream>

int main() {
	system("PAUSE");
}



/*******************************************************************************************
** TESTING  ********************************************************************************
********************************************************************************************/
/*int main()
{	
	int count = -1;
	cudaGetDeviceCount(&count);
	std::cout << "Anzahl devices: " << count << "\n";

	int result = -1;
	initSim(&result);

	int docheck = 0;


	char ** info = (char**)malloc(sizeof(char*)*1);
	info[0] = (char*)malloc(sizeof(char)*255);
	deviceInfo(info);
	printf(*info);
	free(info[0]);
	free(info);
		
	// Test (un)conditional simulation
	{
		float xmin = 0, xmax = 100, ymin = 0, ymax = 100;
		int nx = 4, ny = 4;
		int covmodel = EXP;
		float nugget=0, range=30, sill = 1;
		int k = 5;

		int ret;
		unconditionalSimInit_2f(&xmin,&xmax,&nx,&ymin,&ymax,&ny,&sill,&range,&nugget,&covmodel,&docheck,&ret);
		printf("Errorcode: %i\n",ret);
		float *uncond = (float*)malloc(sizeof(float)*nx*ny*k);
		unconditionalSimRealizations_2f(uncond,&k,&ret);
		printf("Errorcode: %i\n",ret);	
		unconditionalSimRelease_2f(&ret);
		printf("Errorcode: %i\n",ret);	

		// write results to csv file for testing purpose
		for (int l=0; l<k; ++l) {
			std::stringstream ss;
			ss << "C:\\fft\\uncond" << l << ".csv";
			writeCSVMatrix(ss.str().c_str(),uncond + l*nx*ny,nx,ny);		
		}
		free(uncond);

		int numSrc = 10;
		float srcxy[] = {64.56559,83.00241,56.55997,66.50534,31.32781,40.72709,39.21148,57.04371,63.70436,69.80745,49.43141,11.15070,17.64743,62.95207,65.10820,83.82076,74.04069,60.73463,74.86754,73.68782};
		float srcdata[] = {98.98939,103.25515,100.31433,108.50051,78.91263,117.80599,104.18466,105.06300,109.77029,102.14903}; 
		
		float covinv[] = {0.31147267,-0.01321620,-0.01345967,-0.01345882,-0.01339282
						,-0.01346126,-0.01346104,-0.20407215,-0.01342644,-0.01352428
						,0.06730629,-0.01321620,0.17913993,-0.02134759,-0.02135648
						,-0.02474149,-0.02135012,-0.02134977,-0.01323224,-0.02131023
						,-0.02123580,0.10675059,-0.01345967,-0.02134759,0.17823321
						,-0.02178865,-0.02124115,-0.02176937,-0.02176945,-0.01349232
						,-0.02171229,-0.02165272,0.10884684,-0.01345882,-0.02135648
						,-0.02178865,0.17823595,-0.02123981,-0.02176800,-0.02177044
						,-0.01349147,-0.02171093,-0.02165136,0.10883999,-0.01339282
						,-0.02474149,-0.02124115,-0.02123981,0.17935024,-0.02124366
						,-0.02124332,-0.01318273,-0.02139313,-0.02167214,0.10621829
						,-0.01346126,-0.02135012,-0.02176937,-0.02176800,-0.02124366
						,0.17822806,-0.02177159,-0.01349392,-0.02171486,-0.02165528
						,0.10885972,-0.01346104,-0.02134977,-0.02176945,-0.02177044
						,-0.02124332,-0.02177159,0.17822876,-0.01349370,-0.02171451
						,-0.02165493,0.10885797,-0.20407215,-0.01323224,-0.01349232
						,-0.01349147,-0.01318273,-0.01349392,-0.01349370,0.31143185
						,-0.01345864,-0.01351466,0.06746960,-0.01342644,-0.02131023
						,-0.02171229,-0.02171093,-0.02139313,-0.02171486,-0.02171451
						,-0.01345864,0.17834275,-0.02190172,0.10857429,-0.01352428
						,-0.02123580,-0.02165272,-0.02165136,-0.02167214,-0.02165528
						,-0.02165493,-0.01351466,-0.02190172,0.17846290,0.10827642
						,0.06730629,0.10675059,0.10884684,0.10883999,0.10621829
						,0.10885972,0.10885797,0.06746960,0.10857429,0.10827642,-0.54429861};

		
		float *cond = (float*)malloc(sizeof(float)*nx*ny*k);
		int retcode = 0;
		conditionalSimInit_2f(&xmin,&xmax,&nx,&ymin,&ymax,&ny,&sill,&range,&nugget,srcxy,srcdata,&numSrc,covinv, &covmodel, &retcode);
		printf("Errorcode: %i\n",retcode);
		conditionalSimRealizations_2f(cond,&k,&retcode);
		conditionalSimRelease_2f(&retcode);

		// write results to csv file for testing purpose
		for (int l=0; l<k; ++l) {
			std::stringstream ss;
			ss << "C:\\fft\\real" << l << ".csv";
			writeCSVMatrix(ss.str().c_str(),cond + l*nx*ny,nx,ny);		
		}
		free(cond);
	}

	system("PAUSE");
}*/