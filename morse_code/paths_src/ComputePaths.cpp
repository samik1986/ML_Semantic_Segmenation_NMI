//============================================================================
// Name        : ShowCandidates.cpp
// Author      : Dingkang Wang
// Version     :
// Copyright   : Your copyright notice
// Description : Show stats of each edge, there different stats
// 1. Average intensity throughout the edge.
// 2. gradient range (max - min).
// 3. different between it and its pc vector.
//============================================================================

#include "Point.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <set>
using namespace std;

struct Path {

	vector<int> vertices;

	Path() {
		vertices = vector<int>();
	}

	Path(vector<int> _vers) {
		vertices = _vers;
	}

	void println() {
		cout << "Path: ";
		for (auto &i : vertices) {
			cout << i << " ";
		}
		cout << endl;
	}

};

typedef struct Path path;
typedef pair<int, int> iipair;
typedef pair<double, double> ddpair;

unordered_map<Point, int> map_intensity;
unordered_map<Point, double> map_vector;

vector<Point> new_vertices;
vector<iipair> new_edges;


// read in vertices, edges;
void ReadInEdge(string vpath, string epath) {
	new_vertices.clear();
	new_edges.clear();
	cout << "Read in vertices and edges..." << endl;
	ifstream fin;
	int x, y;
	fin.open(vpath.c_str());
	// x y density
	cout << "reading in verts from: " << vpath << endl;
	while (fin >> x >> y) {
		new_vertices.push_back(Point(x, y, 0));
	}
	fin.close();

	int u, v;
	fin.open(epath.c_str());
	// u v i (u, v) starting from 0.
	cout << "reading in edges from: " << epath << endl;
	while (fin >> u >> v) {
		//cout << u << " " << v << endl;
		new_edges.push_back(make_pair(u, v));
	}
	fin.close();
}


void output_paths(string ofilepath, vector<path> &paths) {
	cout << "Output...." << endl;
	ofstream fout;
	fout.open(ofilepath.c_str());
	fout << fixed << setprecision(4);
	for (int i = 0; i < (int) paths.size(); ++i) {
		for(auto id : paths[i].vertices) {
			fout << id << " ";
		}
		fout << endl;
	}

	fout.close();
}


void dfs(unordered_set<int> &used, vector<vector<iipair>> &edgelist, int temp,
                bool start, path &tpath, vector<path> &paths) {

        vector<iipair> edges = edgelist[temp];
        if (!start && edges.size() != 2) {
                // add path;
                paths.push_back(tpath);
        } else {
                for (auto &e : edges) {
                        if (!used.count(e.second)) {
                                used.insert(e.second);
                                tpath.vertices.push_back(e.first);
                                dfs(used, edgelist, e.first, false, tpath, paths);
                                tpath.vertices.pop_back();
                        }
                }
        }
}


vector<path> get_paths() {

	int n = new_vertices.size();
	cout << "new verts: " << n << endl;
	//cout << "t1" << endl;
	vector<vector<iipair>> edgelist = vector<vector<iipair>>(n,
			vector<iipair>());

	//cout << "t2" << endl;
	set<iipair> used_edges;
	for (int i = 0; i < (int) new_edges.size(); ++i) {
		
		//cout << "u1" << endl;
		int u = new_edges[i].first;
		//cout << "u: " << u << endl;
		//cout << "u2" << endl;
		int v = new_edges[i].second;
		//cout << "v: " << v << endl;


		//cout << "u3" << endl;
		if (!used_edges.count(make_pair(min(u, v), max(u, v)))) {
			used_edges.insert(make_pair(min(u, v), max(u, v)));
			edgelist[u].push_back(make_pair(v, i));
			edgelist[v].push_back(make_pair(u, i));
		}
		//cout << "u4" << endl;
	}

	//cout << "t3" << endl;
	unordered_set<int> used;

	vector<path> rets;

	for (int i = 0; i < n; ++i) {
		if (edgelist[i].size() != 2) {
			path p;
			p.vertices.push_back(i);
			vector<path> paths;
			dfs(used, edgelist, i, true, p, paths);
			rets.insert(rets.end(), paths.begin(), paths.end());
		}
	}
	//cout << "t4" << endl;
	return rets;
}

int main(int argc, char **argv) {
		
		//string directory = "../stp_numbered/result/" + to_string(i) + "/";
		string directory = argv[1];
		cout << "working on " << directory << endl;
		string vfile_path = directory + "vert.txt", efile_path = directory + "no_dup_edge.txt"; // input file paths. path to vertice, edge files.
		string oefile_path = directory + "paths.txt";
		ReadInEdge(vfile_path, efile_path);
		cout << "We have " << new_vertices.size() << " vertices, and "
			<< new_edges.size() << " edges." << endl;
		vector<path> paths = get_paths();
		cout << "There are " << paths.size() << " paths." << endl;
		output_paths(oefile_path, paths);

	return 0;
}
