#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

unsigned char *pdata; // pointer to data content

void getInfo(int *width, int *height, int *dataOffset, int *pixLen) {

    FILE *f;

    if (NULL == (f = fopen("lena_color.bmp", "rb"))) {

        printf("Fail to open the file1");

        exit(EXIT_FAILURE);

    }

    fseek(f, 0x00A, SEEK_SET);

    fread(dataOffset, sizeof(char) * 4, 1, f);

    fseek(f, 0x012, SEEK_SET );

    fread(width, sizeof(char) * 4, 1, f);

    fseek(f, 0x016, SEEK_SET);

    fread(height, sizeof(char) * 4, 1, f);

    fseek(f, 0x01C, SEEK_SET);

    fread(pixLen, sizeof(char)* 2, 1, f);
    *pixLen /= 8; //bit to byte
    printf("width = %d, height = %d, dataOffset = %d, pixLen = %d\n", *width, *height, *dataOffset, *pixLen);
    fclose(f);

}

void getData(int width, int height, int dataOffset, int pixLen) {
    FILE *f;

    if (NULL == (f = fopen("lena_color.bmp", "rb"))) {
    	
        printf("Fail to open the file2");

        exit(EXIT_FAILURE);

    }

    fseek(f, dataOffset, SEEK_SET);

    int size = fread(pdata, sizeof(unsigned char), width * height * pixLen, f);
    printf("Data size = %d byte \n", size);

    fclose(f);
}

void copy() {

    FILE *r, *w;

    unsigned char buf[1024];

    if (NULL == (r = fopen("lena_color.bmp", "rb"))) {

        printf("Fail to open the file3");

        exit(EXIT_FAILURE);

    }

    if (NULL == (w = fopen("result.bmp", "wb"))) {

        printf("Fail to open the file4");

        exit(EXIT_FAILURE);

    }

    

    while((fread(buf,sizeof(char),1024,r))>0)
        fwrite(buf,sizeof(char),1024,w);

    fclose(r);

    fclose(w);

}

void writeDataToImg(int width, int height, int dataOffset, int pixLen) {

    FILE *f;

    if (NULL == (f = fopen("result.bmp", "r+b"))) {

        printf("Fail to open the file5");

        exit(EXIT_FAILURE);

    }

    fseek(f, dataOffset, SEEK_SET);

    fwrite(pdata, sizeof(unsigned char), width * height * pixLen, f);

    fclose(f);

}

__global__ void processData(unsigned char *Da, int* filter)
{
    int tx = threadIdx.x;           // thread的x軸id
    int bx = blockIdx.x;            // block的x軸id
    int bn = blockDim.x;  
    int gid = bx * bn + tx;
    __shared__ int sfilter[3][3];
    __shared__ int sR[3][512];      // 每個block存上中下三行
    __shared__ int sG[3][512];
    __shared__ int sB[3][512];
    __shared__ int sRsum[512];      // 每個block 最後512個sum
    __shared__ int sGsum[512];
    __shared__ int sBsum[512];

    if (tx < 9)                     // 每個block 存filter 到 share memory
    {
        sfilter[tx / 3][tx % 3] = filter[tx];
    }
    __syncthreads();

    if (bx == 0 || bx == 511 || tx == 0 || tx == 511)
    {
        // 邊界處理 --> 直接給原本值不動
        sRsum[tx] = Da[gid * 3];
        sGsum[tx] = Da[gid * 3 + 1];
        sBsum[tx] = Da[gid * 3 + 2];
    }

    // 邊界處理(第1個block跟最後一個block不做)
    if (bx != 0 && bx != 511)
    {
    	// R, G, B個別將該Row(Block)運算會用到的上中下三行存入Share Memory
    	sR[0][tx] = Da[gid * 3 - 512 * 3];
		sR[1][tx] = Da[gid * 3];
		sR[2][tx] = Da[gid * 3 + 512 * 3];

		sG[0][tx] = Da[gid * 3 - 512 * 3 + 1];
		sG[1][tx] = Da[gid * 3 + 1];
		sG[2][tx] = Da[gid * 3 + 512 * 3 + 1];

		sB[0][tx] = Da[gid * 3 - 512 * 3 + 2];
		sB[1][tx] = Da[gid * 3 + 2];
		sB[2][tx] = Da[gid * 3 + 512 * 3 + 2];
		__syncthreads();

		// 邊界處理(每個block的的第一個值和最後一個值不做)
		if (tx != 0 && tx != 511)
		{
			// R
			sRsum[tx] = sR[0][tx - 1] * sfilter[0][0];
			sRsum[tx] += sR[0][tx] * sfilter[0][1];
			sRsum[tx] += sR[0][tx + 1] * sfilter[0][2];

			sRsum[tx] += sR[1][tx - 1] * sfilter[1][0];
			sRsum[tx] += sR[1][tx] * sfilter[1][1];
			sRsum[tx] += sR[1][tx + 1] * sfilter[1][2];

			sRsum[tx] += sR[2][tx - 1] * sfilter[2][0];
			sRsum[tx] += sR[2][tx] * sfilter[2][1];
			sRsum[tx] += sR[2][tx + 1] * sfilter[2][2];

			// G
			sGsum[tx] = sG[0][tx - 1] * sfilter[0][0];
			sGsum[tx] += sG[0][tx] * sfilter[0][1];
			sGsum[tx] += sG[0][tx + 1] * sfilter[0][2];

			sGsum[tx] += sG[1][tx - 1] * sfilter[1][0];
			sGsum[tx] += sG[1][tx] * sfilter[1][1];
			sGsum[tx] += sG[1][tx + 1] * sfilter[1][2];

			sGsum[tx] += sG[2][tx - 1] * sfilter[2][0];
			sGsum[tx] += sG[2][tx] * sfilter[2][1];
			sGsum[tx] += sG[2][tx + 1] * sfilter[2][2];

			// B
			sBsum[tx] = sB[0][tx - 1] * sfilter[0][0];
			sBsum[tx] += sB[0][tx] * sfilter[0][1];
			sBsum[tx] += sB[0][tx + 1] * sfilter[0][2];

			sBsum[tx] += sB[1][tx - 1] * sfilter[1][0];
			sBsum[tx] += sB[1][tx] * sfilter[1][1];
			sBsum[tx] += sB[1][tx + 1] * sfilter[1][2];

			sBsum[tx] += sB[2][tx - 1] * sfilter[2][0];
			sBsum[tx] += sB[2][tx] * sfilter[2][1];
			sBsum[tx] += sB[2][tx + 1] * sfilter[2][2];


			sRsum[tx] /= filter[9];
			sGsum[tx] /= filter[9];
			sBsum[tx] /= filter[9];
			// 大於255 或 小於0處理
			if (sRsum[tx] > 255)
				sRsum[tx] = 255;
			else if (sRsum[tx] < 0)
				sRsum[tx] = 0;

			if (sGsum[tx] > 255)
				sGsum[tx] = 255;
			else if (sGsum[tx] < 0)
				sGsum[tx] = 0;

			if (sBsum[tx] > 255)
				sBsum[tx] = 255;
			else if (sBsum[tx] < 0)
				sBsum[tx] = 0;
		}
    }

    __syncthreads();
    
	// 將R, G, B三個陣列值合併寫回一維陣列，以利輸出到檔案
	Da[gid * 3] = sRsum[tx];
	Da[gid * 3 + 1] = sGsum[tx];
	Da[gid * 3 + 2] = sBsum[tx];
}

void ImgDataProcess(int width, int height, int pixLen){

    int DataSize = width * height * pixLen; // 512 * 512 * 3

    /* GPU config */
    unsigned char *Da;
    int f[10];
    int choose;
    // user choose
    printf("請選擇您要的圖片轉換:\n");
    printf("1.模糊化\n");
    printf("2.銳利化\n");
    printf("選擇:");
    scanf("%d", &choose);
   	if (choose == 1)
   	{
   		for (int i = 0;i < 9;i++)
   			f[i] = 1;
   		f[9] = 9; // 模糊化 存最後要除的值
   	}
   	else if (choose == 2)
   	{
   		f[0] = 0; f[1] = -1; f[2] = 0;
   		f[3] = -1; f[4] = 5; f[5] = -1;
   		f[6] = 0; f[7] = -1; f[8] = 0;

   		f[9] = 1; // 銳利化signal
   	}
   	else
   	{
   		printf("沒這選項88");
   		exit(1);
   	}

    int *filter;
    cudaMalloc((void**)&Da, DataSize);          //  create memory for save cpu data in gpu memory 
    cudaMalloc((void**)&filter, 10 * sizeof(int));
    cudaMemcpy(Da, pdata, DataSize, cudaMemcpyHostToDevice);
    cudaMemcpy(filter, f, 10 * sizeof(int), cudaMemcpyHostToDevice);

    //  #########計算時間 宣告
    cudaEvent_t start,stop;      //宣告起始時間和結束時間
    cudaEventCreate(&start);     //分配開始時間的紀錄空間
    cudaEventCreate(&stop);      //分配結束時間的紀錄空間

     /** 開始計時 **/
    cudaEventRecord(start, 0);       //將起始時間歸零並開始計算
    //-------------------
    
    // 處理資料
    dim3 block(512, 1, 1);
    dim3 grid(512, 1, 1);
    processData <<< grid, block >>> (Da, filter);
    cudaThreadSynchronize();
    //-------------------
    
    cudaEventRecord(stop, 0);        //將結束時間歸零並開始計算
    /** 結束計時 **/

    /*time slapsed*/
    cudaEventSynchronize(stop);
    float elaspedTime;
    cudaEventElapsedTime(&elaspedTime, start, stop);
    printf("Exe time: %f\n", elaspedTime); //print time
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    /* #########timing end*/

    // 將資料寫回pdata
    cudaMemcpy(pdata, Da, DataSize, cudaMemcpyDeviceToHost);



    cudaFree(Da);cudaFree(filter);
}

// CPU 
void transfer(int width, int height, int pixLen) {
    int i, j;
    int R[512][512];
    int G[512][512];
    int B[512][512];
    for(i = 0;i < height;i++) {
        for (j = 0;j < width;j++)
        {
            R[i][j] = pdata[(i * width * 3) + (j * 3)];
            G[i][j] = pdata[(i * width * 3) + (j * 3 + 1)];
            B[i][j] = pdata[(i * width * 3) + (j * 3 + 2)];
        }
    }
    int Rsum;
    int Gsum;
    int Bsum;
    for (i = 0;i < height;i++)  
    {
        for (j = 0;j < width;j++)
        {
            Rsum = 0;
            Gsum = 0;
            Bsum = 0;
            if (i == 0 || j == 0 || i == height - 1 || j == width - 1) // 邊緣不處理
            {
                // pdata[(i * width * 3) + (j * 3)] = R[i][j];
                // pdata[(i * width * 3) + (j * 3 + 1)] = G[i][j];
                // pdata[(i * width * 3) + (j * 3 + 2)] = B[i][j];
                continue;
            }
            Rsum += R[i - 1][j - 1] + R[i - 1][j] + R[i - 1][j + 1];
            Rsum += R[i][j - 1] + R[i][j] + R[i][j + 1];
            Rsum += R[i + 1][j - 1] + R[i + 1][j] + R[i + 1][j + 1];

            Gsum += G[i - 1][j - 1] + G[i - 1][j] + G[i - 1][j + 1];
            Gsum += G[i][j - 1] + G[i][j] + G[i][j + 1];
            Gsum += G[i + 1][j - 1] + G[i + 1][j] + G[i + 1][j + 1];


            Bsum += B[i - 1][j - 1] + B[i - 1][j] + B[i - 1][j + 1];
            Bsum += B[i][j - 1] + B[i][j] + B[i][j + 1];
            Bsum += B[i + 1][j - 1] + B[i + 1][j] + B[i + 1][j + 1];
            Rsum /= 9;
            Gsum /= 9;
            Bsum /= 9;
            if (Rsum > 255)
                Rsum = 255;
            else if (Rsum < 0)
                Rsum = 0;
            if (Gsum > 255)
                Gsum = 255;
            else if (Gsum < 0)
                Gsum = 0;
            if (Bsum > 255)
                Bsum = 255;
            else if (Bsum < 0)
                Bsum = 0;
        }
    }
}

int main() {

    int height, width;
    int dataOffset, pixLen;

    getInfo(&width, &height, &dataOffset, &pixLen);

    pdata = (unsigned char *)malloc(sizeof(unsigned char) * height * width * pixLen);

    getData(width, height, dataOffset, pixLen);

	// cpu 版本
	// transfer(width, height, pixLen);

    // 改變原始資料內容(pdata改變)
    ImgDataProcess(width, height, pixLen);

    
    copy(); //copy an backup of "lena.bmp"
    writeDataToImg(width, height, dataOffset, pixLen); // 將資料寫入新圖

    free(pdata);
}