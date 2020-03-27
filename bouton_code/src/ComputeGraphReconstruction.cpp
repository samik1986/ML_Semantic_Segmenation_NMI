#include <opencv2/highgui.hpp>

#include <ctime>
#include <map>
#include <vector>
#include <iostream>
#include <queue>
#include <fstream>

using namespace std;

vector<int> bfs(int vert, int &branch_min, bool visited[], const vector<int> neighborhoods[], int pre[]){
			clock_t d_start = clock();
    	
	queue<int> q;
    	vector<int> visited_list;
	visited_list.clear();
	visited_list.push_back(vert);
	pre[vert] = -1;
    	q.push(vert);

	//cout << "entering queue loop" << endl;
	branch_min = vert;

	int calls = 0;

	while (!q.empty())
	{
		calls++;
		int current_vert = q.front();
		q.pop();
	
		bool is_visited = visited[current_vert];	
		if (!is_visited)
		{
			clock_t d_start = clock();
			visited[current_vert] = true;
			clock_t d_end = clock();
			//cout << "discovered: " << (d_end - d_start) / float(CLOCKS_PER_SEC) << endl;
		}

		if (current_vert < branch_min)
		{
			clock_t m_start = clock();
			branch_min = current_vert;
			clock_t m_end = clock();
			//cout << "min: " << (m_end - m_start) / float(CLOCKS_PER_SEC) << endl;
		}

		vector<int> neighborhood = neighborhoods[current_vert];

		for (int i = 0; i < neighborhood.size(); i++)
		{
			//cout << "working on neighbor " << i << endl;
			int neighbor = neighborhood[i];
			if (!visited[neighbor])
			{
			clock_t m_start = clock();
				visited[neighbor] = true;
				visited_list.push_back(neighbor);
				pre[neighbor] = current_vert;
				q.push(neighbor);
			clock_t m_end = clock();
			//cout << "min: " << (m_end - m_start) / float(CLOCKS_PER_SEC) << endl;
			}
		}
	}
	for (int i = 0; i < visited_list.size(); i++)
	{
		visited[visited_list[i]] = false;
	}
	//cout << "number of calls: " << calls << endl;
			clock_t d_end = clock();
			//cout << "whole call: " << (d_end - d_start) / float(CLOCKS_PER_SEC) << endl;
	return visited_list;
}

void retrieve_path(int vert, vector<int> &vPath, int pre[]){
    vPath.clear();
    //cout << "starting at vert " << vert << endl;
    while(pre[vert] != -1){
        vPath.push_back(vert);
        int vpre = pre[vert];
	//cout << vert << " " << vpre << endl;
        vert = vpre;
    }
    vPath.push_back(vert);
}


int main(int argc, char* argv[])
{
	string input_filename;
	input_filename = argv[1];
	string output_dir = argv[2];
	int persistence_threshold = 8;
	//persistence_threshold = int(argv[2]);
	//string output_dir = "../test_output-7/";

	cout << "Loading image..." << endl;
	cout << input_filename << endl;
	//cv::Mat image = cv::imread(input_filename);
	cv::Mat image = cv::imread(input_filename, cv::IMREAD_ANYDEPTH);
	//cv::Mat image = cv::imread(input_filename, cv::IMREAD_GRAYSCALE);

	if(! image.data ) {
                cout <<  "Could not open or find the image" << endl;
                return -1;
        } 

	//image = image(cv::Rect(172,275,13,10));

	cout << "Image loaded" << endl;
	
	int rows = image.rows;
	int cols = image.cols;

	vector<vector<int> > verts;
	verts.clear();
	int v_index = 0;
	cout << "Reading in verts" << endl;
	for (int j = 0; j < cols; j++)
	{
		for (int i = 0; i < rows; i++)
		{
			vector<int> vert;
			vert.clear();
			vert.push_back(v_index);
			vert.push_back(i);
			vert.push_back(j);
			/*
			if ((int) image.at<uchar>(i, j) == 85)
			{
				cout << "GOT IT" << endl;
			}
			*/
			cout << "Value at (" << i << ", " << j << "): " << (int) image.at<unsigned short>(i, j) << endl;
			vert.push_back(-image.at<unsigned short>(i, j));
			//vert.push_back(- ((int) image.at<uchar>(i, j)));
			verts.push_back(vert);
			v_index++;
		}
	}

	/*
	for (int i = 0; i < verts.size(); i++)
	{
		cout << verts[i][0] << ' ' << verts[i][1] << ' ' << verts[i][2] << ' ' << verts[i][3] << endl;
	}
	*/
	
	cout << "read in " << verts.size() << " verts" << endl;	
	cout << "sorting verts by function value..." << endl;

	sort(verts.begin(), verts.end(), [](const vector<int>& a, const vector<int>& b)
			{ 
				if (a[3] == b[3])
				{
					return a[0] < b[0];
				}
				
				return a[3] < b[3]; 
			});
	/*
	for (int i = 0; i < verts.size(); i++)
	{
		cout << verts[i][0] << ' ' << verts[i][1] << ' ' << verts[i][2] << ' ' << verts[i][3] << endl;
	}
	*/
	
	cout << "creating vertex dictionary" << endl;
	map<int, int> vertex_filtration_dict;
	for (int i = 0; i < verts.size(); i++)
	{
		int index = verts[i][0];
		vertex_filtration_dict[index] = i;
	
	}
	
	/*
	for (int i = 0; i < verts.size(); i++)
	{
		cout << i << ": " << verts[i][0] << endl;
	}
	*/
	// return 1;
	
	//calculating faces
	cout << "calculation faces..." << endl;
	vector<vector<int> > faces;
	faces.clear();
	int f_index = 0;
	for (int i = 0; i < rows - 1; i++)
	{
		for (int j = 0; j < cols - 1; j++)
		{
			int v0_id = j * rows + i;
			int v1_id = v0_id + 1;
			int v2_id = v0_id + rows;
			int v3_id = v2_id + 1;

			// cout << v0_id << ' ' << v1_id << ' ' << v2_id << ' ' << v3_id << endl;

			int v0 = vertex_filtration_dict[v0_id];
			int v1 = vertex_filtration_dict[v1_id];
			int v2 = vertex_filtration_dict[v2_id];
			int v3 = vertex_filtration_dict[v3_id];

			//cout << v0 << ' ' << v1 << ' ' << v2 << ' ' << v3 << endl;
			
			vector<int> face;
			face.clear();
			face.push_back(v0);
			face.push_back(v1);
			face.push_back(v2);
			face.push_back(v3);
			
			//cout << face[0] << ' ' << face[1] << ' ' << face[2] << ' ' << face[3]  << endl;
			sort(face.begin(), face.end(), [](int a, int b)
                        {
                                return a < b;
                        });

			face.push_back(f_index);
			//cout << face[0] << ' ' << face[1] << ' ' << face[2] << ' ' << face[3] << ' ' << face[4] << endl;
			faces.push_back(face);
			f_index++;
		}
	}

	/*
	for (int i = 0; i < faces.size(); i++)
	{
		cout << faces[i][0] << ' ' << faces[i][1] << ' ' << faces[i][2] << ' ' << faces[i][3] << ' ' << faces[i][4] << endl;
	}
	*/
	//return 1;

	//sort faces
	cout << "sorting faces" << endl;
	sort(faces.begin(), faces.end(), [](const vector<int>& a, const vector<int>& b)
                        {
                                if (a[3] == b[3] and a[0] == b[0] and a[1] == b[1])
                                {
                                        return a[2] < b[2];
                                }

				if (a[3] == b[3] and a[0] == b[0])
                                {
                                        return a[1] < b[1];
                                }

				if (a[3] == b[3])
                                {
                                        return a[0] < b[0];
                                }

                                return a[3] < b[3];
                        });
	/*
	for (int i = 0; i < faces.size(); i++)
	{
		cout << faces[i][0] << ' ' << faces[i][1] << ' ' << faces[i][2] << ' ' << faces[i][3] << ' ' << faces[i][4] << endl;
	}
	*/
	//return 1;
	
	cout << "building face dict" << endl;
	map<int, int> face_filtration_dict;
	for (int i = 0; i < faces.size(); i++)
	{
		int index = faces[i][4];
		face_filtration_dict[index] = i;
	}
	/*
	for (int i = 0; i < faces.size(); i++)
	{
		cout << i << ": " << faces[i][4] << endl;
	}
	*/
	// return 1;

	cout << "calculating edges" << endl;
	vector<vector<int> > edges;
	edges.clear();
	vector<vector<int> > face_edges;
	face_edges.clear();

	int max_func_val = verts[verts.size() - 1][3];
	int n_faces = faces.size();
	int adjust = (rows * cols) % 2;
	//horizontal edges
	
	int edge_counter = 0;
	int face_index = 0;
	for (int i = 0; i < rows - 1; i++)
	{
		for (int j = 0; j < cols; j++)
		{
		       	vector<int> edge;
			edge.clear();
			int v0 = vertex_filtration_dict[j * rows + i];
			int v1 = vertex_filtration_dict[j * rows + i + 1]; 
			if (v0 < v1)
			{
				edge.push_back(v0);
				edge.push_back(v1);	
			}
			else
			{
				edge.push_back(v1);
				edge.push_back(v0);
			}

			edge.push_back(edge_counter);

			//cout << verts[edge[0]][0] << ' ' << verts[edge[1]][0] << endl;
			edges.push_back(edge);
			
			
			vector<int> face_edge;
			face_edge.clear();
			int boundary_check = edge_counter % cols;
			// upper boundary
			if (boundary_check == 0)
			{
				face_edge.push_back(face_filtration_dict[i * cols - i]);
				face_edge.push_back(n_faces);
			}
			//lower boundary
			else if (boundary_check == cols - 1)
			{
				//face_edge.push_back(face_filtration_dict[i * cols + cols - 1]);
				face_edge.push_back(face_filtration_dict[face_index]);
				face_edge.push_back(n_faces);
				face_index++;
			}
			else
			{
				
				face_edge.push_back(face_filtration_dict[face_index]);
				face_edge.push_back(face_filtration_dict[face_index + 1]);
				face_index++;
			}
			
			sort(face_edge.begin(), face_edge.end(), [](int a, int b)
                        {
                                return a < b;
                        });
			
			face_edges.push_back(face_edge);
			// cout << face_edge[0] << ", " << face_edge[1] << endl;
			
			edge_counter++;
		}
	}
	
	cout << "vertical edges..." << endl;
	//vertical edges
	for (int j = 0; j < cols - 1; j++)
	{
		for (int i = 0; i < rows; i++)
		{
			vector<int> edge;
			edge.clear();
			int v0 = vertex_filtration_dict[j * rows + i];
			int v1 = vertex_filtration_dict[j * rows + i + rows];
			// cout << j * rows + i << ", " << j * rows + i + rows << endl;
			if (v0 < v1)
			{
				edge.push_back(v0);
				edge.push_back(v1);
			}
			else
			{
				edge.push_back(v1);
				edge.push_back(v0);
			}
			edge.push_back(edge_counter);
			//cout << verts[edge[0]][0] << ' ' << verts[edge[1]][0] << endl;
			edges.push_back(edge);

			vector<int> face_edge;
			face_edge.clear();
			
			int boundary_check = i % rows;
			if (boundary_check == 0)
			{
				//left boundary
				//cout << "left: " << i << " " << j << endl;
				face_edge.push_back(face_filtration_dict[j]);
				face_edge.push_back(n_faces);
			}
			else if (boundary_check == rows - 1)
			{
				//right boundary
				//cout << "right: " << i << " " << j << endl;
				//cout << j + (i - 1) * (cols - 1) << ": " << face_filtration_dict[j + (i - 1) * (cols - 1)] << endl;
				face_edge.push_back(face_filtration_dict[j + (i - 1) * (cols - 1)]);
				face_edge.push_back(n_faces);
				
			}
			else
			{
				//cout << "mid: " << i << " " << j << endl;
				//cout << j + (i - 1) * (cols - 1) << ": " << face_filtration_dict[j + (i - 1) * (cols - 1)] << endl;
				//cout << j + i * (cols - 1) << ": " << face_filtration_dict[j + i * (cols - 1)] <<  endl;
				face_edge.push_back(face_filtration_dict[j + (i - 1) * (cols - 1)]);
				face_edge.push_back(face_filtration_dict[j + i * (cols - 1)]);
			}
			
			sort(face_edge.begin(), face_edge.end(), [](int a, int b)
                        {
                                return a < b;
                        });
			
			
			face_edges.push_back(face_edge);
			// cout << face_edge[0] << ", " << face_edge[1] << endl;
			edge_counter++;
		}
	}

	//sort edges
	cout << "sorting edges by filtration" << endl;
	sort(edges.begin(), edges.end(), [](const vector<int>& a, const vector<int>& b)
                        {
                                if (a[1] == b[1])
                                {
                                        return a[0] < b[0];
                                }

                                return a[1] < b[1];
                        });
	/*
        for (int i = 0; i < edges.size(); i++)
        {
                cout << edges[i][0] << " " << edges[i][1] << endl;
        }
	*/
	cout << "building edge dict" << endl;
        map<int, int> edge_filtration_dict;
        for (int i = 0; i < edges.size(); i++)
        {
		vector<int> edge = edges[i];
                int index = edge[2];
                edge_filtration_dict[index] = i;
        }

	/*
        for (int i = 0; i < edges.size(); i++)
        {
                cout << i << ": " << edges[i][2] << endl;
        }
	

	cout << "face edge check" << endl;
	for (int i = 0; i < face_edges.size(); i++)
	{
                cout << i << ": " << face_edges[edges[i][2]][0] << " "  << face_edges[edges[i][2]][1] << endl;
	}
	*/

	// return 1;

	cout << "setting up union find for minimum spanning tree" << endl;
	vector<int> vert_ptr;
	vert_ptr.clear();
	
	vector<vector<int> > vert_component;
	vert_component.clear();
	for (int i = 0; i < verts.size(); i++)
	{
		vert_ptr.push_back(i);
		vector<int> component;
		component.clear();
		component.push_back(i);
		vert_component.push_back(component);
	}
	
	vector<int> ve_persistence;
	ve_persistence.clear();
	vector<int> vert_persistence;
	vert_persistence.clear();
	for (int i = 0; i < verts.size(); i++)
	{
		vert_persistence.push_back(-1);
	}
	int errors = 0;
	cout << "computing minimum spanning tree" << endl;
	for (int i = 0; i < edges.size(); i++)
	{
		vector<int> edge = edges[i];
		//cout << "working on edge " << edge[2] << endl;
		int v0 = edge[0];
		int v1 = edge[1];

		int edge_func_val;
		if (verts[v0][3] > verts[v1][3])
		{
			edge_func_val = verts[v0][3];
		}
		else
		{
			edge_func_val = verts[v1][3];
		}

		//cout <<"function value = " << edge_func_val << endl;

		int v0_rep = vert_ptr[v0];
		int v1_rep = vert_ptr[v1];
		if (v0_rep == v1_rep)
		{
			ve_persistence.push_back(-1);
			continue;
		}
		
		int v0_rep_val = verts[v0_rep][3];
		int v1_rep_val = verts[v1_rep][3];

		if (v0_rep < v1_rep)
		{
			ve_persistence.push_back(edge_func_val - v1_rep_val);
			vector<int> comp = vert_component[v1_rep];
			if (vert_persistence[v1_rep] != -1)
			{
				cout << "error at " << v1 << endl;
				errors++;
			}
			vert_persistence[v1_rep] = edge_func_val - v1_rep_val;
			for (int j = 0; j < comp.size(); j++)
			{
				vert_ptr[comp[j]] = v0_rep;
				vert_component[v0_rep].push_back(comp[j]);
			}
		}
		else
		{
			ve_persistence.push_back(edge_func_val - v0_rep_val);
			vector<int> comp = vert_component[v0_rep];
			if (vert_persistence[v0_rep] != -1)
			{
				errors++;
			}
			vert_persistence[v0_rep] = edge_func_val - v0_rep_val;
			for (int j = 0; j < comp.size(); j++)
			{
				vert_ptr[comp[j]] = v1_rep;
				vert_component[v1_rep].push_back(comp[j]);
			}
		}
	}

	int pos_ve_count = 0;	
	for (int i = 0; i < ve_persistence.size(); i++)
	{
		if (ve_persistence[i] >= 0)
		{
			//cout << ve_persistence[i] << endl;
			//cout << verts[edges[i][0]][0] << ' ' << verts[edges[i][1]][0] << endl;
			pos_ve_count++;
		}
	}
	cout << pos_ve_count << '/' << ve_persistence.size() << endl;
	cout << "test ERROR: " << errors << endl;

	/*
	for (int i = 0; i < ve_persistence.size(); i++)
	{
		cout << ve_persistence[i] << " ";
	}
	*/
	
	// return 1;

	cout << "setting up union find for maximum spanning tree" << endl;
        vector<int> face_ptr;
        face_ptr.clear();

        vector<vector<int> > face_component;
        face_component.clear();
	
	// <= to include face that represents faces outside our domain that are part of dual
        for (int i = 0; i <= faces.size(); i++)
        {
                face_ptr.push_back(i);
                vector<int> component;
                component.clear();
                component.push_back(i);
                face_component.push_back(component);
        }


        vector<int> et_persistence;
        et_persistence.clear();
	cout << "computing maximum spanning tree in dual" << endl;
        for (int i = edges.size() - 1; i >= 0; i--)
        {
                vector<int> edge = edges[i];
                int v0 = edge[0];
                int v1 = edge[1];
		int edge_func_val;
		if (verts[v0][3] > verts[v1][3])
		{
			edge_func_val = verts[v0][3];
		}
		else
		{
			edge_func_val = verts[v1][3];
		}
		//cout << "test" << endl;
		int edge_index = edge[2];
		vector<int> face_edge = face_edges[edge_index];
		int f0 = face_edge[0];
		int f1 = face_edge[1];


                int f0_rep = face_ptr[f0];
                int f1_rep = face_ptr[f1];

                
		if (f0_rep == f1_rep)
                {
                        et_persistence.push_back(-1);
                        continue;
                }
		
		//cout << "test" << endl;

		// get the face
                int f0_val;
		if (f0_rep == n_faces)
		{
			f0_val = max_func_val;
		}
		else
		{
			f0_val = verts[faces[f0_rep][3]][3];
		}

		int f1_val;
		if (f1_rep == n_faces)
		{
			f1_val = max_func_val;
		}
		else
		{
			f1_val = verts[faces[f1_rep][3]][3];
		}

		//cout << "face vals: " << f0_val << " " << f1_val << endl;

                if (f0_rep > f1_rep)
                {
                        et_persistence.push_back(f1_val - edge_func_val); // edge_func_val - v1_rep_val);
                        vector<int> comp = face_component[f1_rep];
                        for (int j = 0; j < comp.size(); j++)
                        {
                                face_ptr[comp[j]] = f0_rep;
                                face_component[f0_rep].push_back(comp[j]);
                        }
		}
                else
                {
                        et_persistence.push_back(f0_val - edge_func_val);
                        vector<int> comp = face_component[f0_rep];
                        for (int j = 0; j < comp.size(); j++)
                        {
                                face_ptr[comp[j]] = f1_rep;
                                face_component[f1_rep].push_back(comp[j]);
                        }
                }
        }

	reverse(et_persistence.begin(), et_persistence.end());

	
	int pos_et_count = 0;	
	for (int i = 0; i < et_persistence.size(); i++)
	{
		if (et_persistence[i] >= 0)
		{
			//cout << et_persistence[i] << endl;
			//cout << verts[edges[i][0]][0] << ' ' << verts[edges[i][1]][0] << endl;
			pos_et_count++;
		}
	}
	cout << pos_et_count << '/' << et_persistence.size() << endl;

	for (int i = 0; i < edges.size(); i++)
	{
		int ve_per = ve_persistence[i];
		int et_per = et_persistence[i];
		if (ve_per == -1 and et_per == -1)
		{
			cout << "Edge " << i << " is not matched at all" << endl;
			return -1;
		}
		else if (ve_per >= 0 and et_per >= 0)
		{
			cout << "Edge " << i << " is paired with both an edge and a triangle" << endl;
			return -1;
		}
	}

	if (pos_ve_count + pos_et_count != edges.size())
	{
		cout << "assignmented dont add up" << endl;
		return -1;
	}

	/*
	for (int i = 0; i < et_persistence.size(); i++)
	{
		cout << et_persistence[i] << " ";
	}
	cout << endl;
	*/
	//return 1;

	cout << "Computing vector field" << endl;
	vector<int> neighborhoods[verts.size()];
	// neighborhoods.clear();
	for (int i = 0; i < verts.size(); i++)
	{
		vector<int> neighborhood;
		neighborhood.clear();
		neighborhoods[i] = neighborhood;

	}

	
        vector<vector<int> > vf_edges;
        vf_edges.clear();
	int ve_in_vf = 0;
	for (int i = 0; i < edges.size(); i++)
	{
		int persistence = ve_persistence[i];
		if (persistence > persistence_threshold or persistence == -1)
		{
			continue;
		}
		
		ve_in_vf++;
		//cout << i << endl;

		vector<int> edge = edges[i];
		int v0 = edge[0];
		int v1 = edge[1];
		vector<int> field_edge;
		field_edge.clear();
		field_edge.push_back(v0);
		field_edge.push_back(v1);
		vf_edges.push_back(field_edge);
		neighborhoods[v0].push_back(v1);
		neighborhoods[v1].push_back(v0);
	}

	cout << "ve in vector field: " << ve_in_vf << endl;
	cout << "vector_field_edges: " << vf_edges.size() << endl;
	
	/*
	for (int i = 0; i < neighborhoods.size(); i++)
	{
		//cout << i << " neighbors: ";
		for (int j = 0; j < neighborhoods[i].size(); j++)
		{
			cout << neighborhoods[i][j] << " ";
		}
		cout << endl;
	}
	*/

	// return 1;

	cout << "Computing manifold" << endl;
	vector<int> min_computed;
	min_computed.clear();

	bool visit[verts.size()];

	int next_in_path[verts.size()];

	vector<vector<int >> manifold;
	manifold.clear();

	for (int i = 0; i < verts.size(); i++)
	{
		min_computed.push_back(-1);
		visit[i] = false;
		next_in_path[i] = -1;
	}
	

	int know_min = 0;
	int not_know_min = 0;
	int critical_count = 0;
	for (int i = 0; i < edges.size(); i++)
	{
		int ve_per = ve_persistence[i];
		int et_per = et_persistence[i];
		
		int persistence;
		if (ve_persistence[i] >= 0)
		{
			persistence = ve_per;
		}
		else
		{
			persistence = et_per;
		}
		

		//cout << "persistence: " << ve_persistence[i] << ' ' << et_persistence[i] << endl;
		//int persistence = ve_per;
		if (persistence <= persistence_threshold)
		{
			continue;
		}

		critical_count++;
		/*
		if (critical_count != 2)
		{
			continue;
		}
		*/

		// cout << "working on critical edge " << critical_count << endl;
		clock_t start = clock();
		vector<int> edge = edges[i];
		vector<int> critical_edge;
		critical_edge.clear();
		critical_edge.push_back(edge[0]);
		critical_edge.push_back(edge[1]);
		manifold.push_back(critical_edge);

		for (int j = 0; j < edge.size() - 1; j++)
		{
			//cout << "working on " << j << "th vert" << endl;
			int v = edge[j];
			vector<int> vPath;
			vPath.clear();
			if (min_computed[v] == -1)
			{
				not_know_min++;
				//cout << "have not computed yet" << endl;
				int branch_min;
				vector<int> component = bfs(v, branch_min, visit, neighborhoods, next_in_path);
				for (int k = 0; k < component.size(); k++)
				{
					min_computed[component[k]] = branch_min;
				}

				//cout << "component size: " << component.size() << endl;
				//cout << "minimum: " << branch_min << endl;
				bfs(branch_min, branch_min, visit, neighborhoods, next_in_path);
				retrieve_path(v, vPath, next_in_path);
			}
			else
			{
				know_min++;
				retrieve_path(v, vPath, next_in_path);
			}
			manifold.push_back(vPath);
		}
		clock_t end = clock();
		//cout << "time for iteration: " << CLOCKS_PER_SEC << endl;
	}

	cout << "critical edges: " << critical_count << endl;
	cout << "not know min: " << not_know_min << endl;
	cout << "know min: " << know_min << endl;

	/*
	for (int i = 0; i < manifold.size(); i++)
        {
                cout << i << " path: ";
                for (int j = 0; j < manifold[i].size(); j++)
                {
                        cout << manifold[i][j] << " ";
                }
                cout << endl;
        }
	*/

	//return 1;

	cout << "outputting..." << endl;
	
	vector<int> output_indices;
	output_indices.clear();
	for (int i = 0; i < verts.size(); i++)
	{
		output_indices.push_back(-1);
	}

	int output_index = 0;
	vector<int> output_verts;
	output_verts.clear();
	vector<int> output_persistence;
	output_persistence.clear();
	vector<vector<int> > output_edges;
	output_edges.clear();
	for (int i = 0; i < manifold.size(); i++)
	{
		vector<int> component = manifold[i];
        /*
        for (int i = 0; i < vf_edges.size(); i++)	
	{
		vector<int> component = vf_edges[i];
	*/
	  	for (int j = 0; j < component.size() - 1; j++)
                {
			//cout << "beginning of i loop: " << output_index << endl;
                        int v0 = component[j];
			int ov0;
			if (output_indices[v0] != -1)
			{
				ov0 = output_indices[v0];
			}
			else
			{
				ov0 = output_index;
				output_indices[v0] = output_index;
				//cout << output_index << ": " << v0 << endl;
				output_verts.push_back(v0);
				output_persistence.push_back(vert_persistence[v0]);
				output_index = output_index + 1;
			}
			//cout << "after v0: " << output_index << endl;

			int v1 = component[j + 1];
                        int ov1;
                        if (output_indices[v1] != -1)
                        {
				//cout <<"t1" << endl;
                                ov1 = output_indices[v1];
                        }
                        else
                        {
				//cout << ov1 << " ";
                                ov1 = output_index;
				//cout << ov1 << " ";
                                output_indices[v1] = output_index;
				//cout << output_indices[v1] << endl;
				//cout << output_index << ": " << v1 << endl;
				output_verts.push_back(v1);
				output_persistence.push_back(vert_persistence[v1]);
                                output_index = output_index + 1;
                        }
			
			vector<int> edge;
			edge.clear();
			edge.push_back(ov0);
			edge.push_back(ov1);
			output_edges.push_back(edge);
                }
	}

	cout << "writing files" << endl;

	
	string vertex_filename = output_dir + "vert.txt";
	ofstream vFile(vertex_filename);
       	for (int i = 0; i < output_verts.size(); i++)
	{
		vector<int> vert = verts[output_verts[i]];
		int persistence = output_persistence[i];
		vFile << vert[1] << " " << vert[2] << " " << persistence << endl;
	}

	string edge_filename = output_dir + "edge.txt";
	ofstream eFile(edge_filename);
       	for (int i = 0; i < output_edges.size(); i++)
	{
		vector<int> edge = output_edges[i];
		//cout << edge[0] << " " << edge[1] << endl;
		eFile << edge[0] << " " << edge[1] << endl;
	}
	
	/*	
	for (int i = 0; i < verts.size(); i++)
	{
		if (output_indices[i] == -1)
		{
			continue;
		}
		cout << i << ": " << output_indices[i] << endl;
	}

	cout << output_index << endl;
	*/

	return 0;
}
