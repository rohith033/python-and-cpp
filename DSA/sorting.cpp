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
    
    
    
int main(){
}
  

  
