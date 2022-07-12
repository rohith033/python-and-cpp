#include<bits/stdc++.h>
using namepsace std;
template<typename T>
int const size_ = 1e6;
T parent[size_];
void make_set(T a){
  parent[a] = a;
}
void find_set(T a)
{
if(a==parent[a]) return a;
else {
return find_set[parent[a]];
}}
void group_pairs(int a,int b){
  T x , y = find_set(a) , find_set(b);
  if(x!=y) parent[b] = a;
}
void Dsu(T arr[]; int size_){
}
int main(){}
  
