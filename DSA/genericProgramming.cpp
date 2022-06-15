// to aviod functional over loading and to make use of single function for different data types generic programming can be used 
// in c++ using templates we achive generic programming 
/*
syntax of template 
    template<typename t> 
    t function_name(){}
    int main(){
    function_name<dataType>(input);
}
*/
// simple sort function
// growth of a generic function in a given binary hex tree will always be a polynomial number in sucha case if we sort the tree by using any inplace alogoritm we will be using extensive memory
#include<bits/stdc++.h>
using namespace std;
template<typename T>
void allsort(T arr[],int size_){
 
  int i, j;
    for (i = 0; i < size_- 1; i++)
        // Last i elements are already
        // in place
        for (j = 0; j < size_ - i - 1; j++)
            if (arr[j] > arr[j + 1])
                swap(arr[j], arr[j + 1]);
}
int main()
{
    char a[5]={'a','A','d','e','f'};
    allsort<char>(a,5);
    for(int i=0;i<5;i++) cout<<a[i]<<endl;
    int arr[5]={5,4,3,2,1};
    allsort<int>(arr,5);
    for(int i=0;i<5;i++) cout<<arr[i]<<endl;
   //if('a'>'c') printf("error");
  return 0;
}
