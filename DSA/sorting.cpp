// sorting :
#include<bits/stdc++.h>
using namespace std;

// selection sort : sort the array index wise:
//                   run a double for loop from i=0 to n and j = i to n and replace the the i th minimum element in ith positiom
                  
template<typename> T;
void selectionsort(T arr[],int size_){
 for(int i=0;i<size_-1;i++)
 {
   for(int j=i+1;j<size_-1;j++){
      if(arr[i]>arr[j]) {
        swap(arr[i],arr[j]);
      }
   }
 }
}
// bubble sort : maximum element bubbles up
//    checking wheather two numbers are in correct order or not if not we would be swaping them by this way we would be setting last element correcty after 1 full cycle 
void bubblesort(T arr[],int size_)
   {
     for(int i=0;i<size_;i++){
       for(int j=0;i<size_-(i+1);j++){
         if(arr[j]>arr[j+1]) swap(arr[j+1],arr[j]);
     }
   }
// insertion sort : placing in correct indexs 
  void insertionsort(T arr[],int size_)
{
  for(int i=1;i<size_;i++){
   for(j=i-1;j>=0;j--){
     if(arr[j]>arr[i]){
       swap(arr[i],arr[j]);
     }
     else break;
   }
}
    
  void seive(int n){
    int primes[n+1];
    memset(primes,true,n+1);
    for(int i=2;i*i<=n;i++){
      for(int j=i*i;j<n;j+=i){
        primes[j]=false;}
      for(int i=0;i<n+1;i++){
        cout<<i<<" "<<":"<<primes[i]<<endl;
      }
  }
    // out of o(n^2) algos insersort is better for large data sets if we have to write selection sort is better as selection sort will do only o(n) swaps where as insertion sort will be doing o(n^2) swaps
    // bubble sort is usally not good in comprasin to these two algos 
    
    //merge sort if size is 1,0 the array is sorted else weh have to call morge sort again and again
    void merge(T arr[],int start,int mid,int end){
      int idx1=0;
      int idx2=0;
      T* temp1 = new T[(mid-start)+1]; // creat two temp arrays and replace the two input array into these temp arrays
      T* temp2 = new T[(end-mid)];
      for(int i=0;i<((mid-start)+1);i++){
        temp1[i]=arr[i];
      }
      for(int i=0;i<(end-mid);i++){
        temp2[i]=arr[(mid-start)+(i+1)];
      }
      int idx=0;
      while((idx1<((mid-start)+1))&&idx2<((end-mid))){
        if(temp1[idx1]>temp2[idx2]){
          arr[idx]=temp2[idx2];
          idx2++;
          idx++;
        }
         else{
           arr[idx]=temp1[idx1];
          idx1++;
          idx++;
         }
        
      }
      while((idx1<((mid-start)+1))){
         arr[idx]=temp1[idx1];
          idx1++;
          idx++;
      }
       while((idx2<(end - mid))){
         arr[idx]=temp1[idx2];
          idx2++;
          idx++;
      }
    }
    void mergesort(T arr[],int start,int end){
      if(size_<1) return;
      mid=start+((begin-start)/2);
      mergesort(arr,start,mid);
      mergesort(arr,mid+1,end);
      merge(arr,start,mid,end);
    }
    
    
int main(){
  int arr[5]={234,45287,5542,255,1};
  mergesort<int>(arr,0,4);
  for(auto i:arr) cout<<i<<endl;
}
  

  
